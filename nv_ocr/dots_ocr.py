import os
import re
import json
import glob
import argparse
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

from nv_ocr.model.inference import inference_with_vllm
from nv_ocr.utils.consts import image_extensions, MIN_PIXELS, MAX_PIXELS
from nv_ocr.utils.image_utils import fetch_image, smart_resize, get_image_by_fitz_doc
from nv_ocr.utils.prompts import dict_promptmode_to_prompt
from nv_ocr.utils.layout_utils import post_process_output, draw_layout_on_image, pre_process_bboxes
from nv_ocr.utils.format_transformer import layoutjson2md


def extract_images_from_text(text):
    """
    Extract image references from markdown text and return cleaned text with image data.
    
    Args:
        text (str): Input markdown text containing image references
    
    Returns:
        dict: Dictionary containing:
            - 'cleaned_text': Text with image references removed
            - 'images': List of extracted image information
            - 'image_count': Number of images found
    """
    # Regex pattern to match markdown image syntax: ![](something)
    image_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)', re.DOTALL)
    
    # Find all image references
    matches = image_pattern.findall(text)
    
    # Create cleaned content by removing image references
    cleaned_text = image_pattern.sub('', text)
    
    # Remove extra empty lines that might be left after removing images
    cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

class ImagesToOCRConverter:
    """
    Convert image directories to OCR results (markdown/json files)
    """
    
    def __init__(self, 
                 ip='localhost',
                 port=8000,
                 model_name='model',
                 temperature=0.1,
                 top_p=1.0,
                 max_completion_tokens=16384,
                 num_thread=64,
                 dpi=200,
                 output_dir="./ocr_output",
                 min_pixels=None,
                 max_pixels=None):
        
        # VLLM server parameters
        self.ip = ip
        self.port = port
        self.model_name = model_name
        
        # Inference parameters
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens
        
        # Processing parameters
        self.num_thread = num_thread
        self.dpi = dpi
        self.output_dir = output_dir
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        
        # Validate pixel constraints
        assert self.min_pixels is None or self.min_pixels >= MIN_PIXELS
        assert self.max_pixels is None or self.max_pixels <= MAX_PIXELS

    def _inference_with_vllm(self, image, prompt):
        """Call VLLM inference"""
        response = inference_with_vllm(
            image,
            prompt, 
            model_name=self.model_name,
            ip=self.ip,
            port=self.port,
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=self.max_completion_tokens,
        )
        return response

    def get_prompt(self, prompt_mode, bbox=None, origin_image=None, image=None, min_pixels=None, max_pixels=None):
        """Get prompt based on mode"""
        prompt = dict_promptmode_to_prompt[prompt_mode]
        if prompt_mode == 'prompt_grounding_ocr':
            assert bbox is not None
            bboxes = [bbox]
            bbox = pre_process_bboxes(origin_image, bboxes, input_width=image.width, input_height=image.height, min_pixels=min_pixels, max_pixels=max_pixels)[0]
            prompt = prompt + str(bbox)
        return prompt

    def _process_single_image(self, image_path, prompt_mode, save_dir, save_name, bbox=None, fitz_preprocess=False):
        """Process a single image"""
        min_pixels, max_pixels = self.min_pixels, self.max_pixels
        
        if prompt_mode == "prompt_grounding_ocr":
            min_pixels = min_pixels or MIN_PIXELS
            max_pixels = max_pixels or MAX_PIXELS
            
        if min_pixels is not None: 
            assert min_pixels >= MIN_PIXELS, f"min_pixels should >= {MIN_PIXELS}"
        if max_pixels is not None: 
            assert max_pixels <= MAX_PIXELS, f"max_pixels should <= {MAX_PIXELS}"

        # Load and preprocess image
        origin_image = fetch_image(image_path)
        
        if fitz_preprocess:
            image = get_image_by_fitz_doc(origin_image, target_dpi=self.dpi)
            image = fetch_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
        else:
            image = fetch_image(origin_image, min_pixels=min_pixels, max_pixels=max_pixels)
            
        input_height, input_width = smart_resize(image.height, image.width)
        
        # Get prompt and run inference
        prompt = self.get_prompt(prompt_mode, bbox, origin_image, image, min_pixels=min_pixels, max_pixels=max_pixels)
        response = self._inference_with_vllm(image, prompt)
        
        result = {
            'image_path': image_path,
            'save_name': save_name,
            "input_height": input_height,
            "input_width": input_width
        }
        
        # Process results based on prompt mode
        if prompt_mode in ['prompt_layout_all_en', 'prompt_layout_only_en', 'prompt_grounding_ocr']:
            cells, filtered = post_process_output(
                response, 
                prompt_mode, 
                origin_image, 
                image,
                min_pixels=min_pixels, 
                max_pixels=max_pixels,
            )
            
            if filtered and prompt_mode != 'prompt_layout_only_en':
                # Model output JSON failed, use filtered process
                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, 'w', encoding="utf-8") as w:
                    json.dump(response, w, ensure_ascii=False)

                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                origin_image.save(image_layout_path)
                
                md_file_path = os.path.join(save_dir, f"{save_name}.md")
                with open(md_file_path, "w", encoding="utf-8") as md_file:
                    md_file.write(cells)
                    
                result.update({
                    'layout_info_path': json_file_path,
                    'layout_image_path': image_layout_path,
                    'md_content_path': md_file_path,
                    'filtered': True
                })
            else:
                # Normal processing
                try:
                    image_with_layout = draw_layout_on_image(origin_image, cells)
                except Exception as e:
                    print(f"Error drawing layout on image: {e}")
                    image_with_layout = origin_image

                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, 'w', encoding="utf-8") as w:
                    json.dump(cells, w, ensure_ascii=False)

                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                image_with_layout.save(image_layout_path)
                
                result.update({
                    'layout_info_path': json_file_path,
                    'layout_image_path': image_layout_path,
                })
                
                if prompt_mode != "prompt_layout_only_en":
                    md_content_with_imgs = layoutjson2md(origin_image, cells, text_key='text')
                    md_content = extract_images_from_text(md_content_with_imgs)
                    md_content_no_hf = layoutjson2md(origin_image, cells, text_key='text', no_page_hf=True)
                    
                    md_file_path = os.path.join(save_dir, f"{save_name}.md")
                    with open(md_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content)
                        
                    md_nohf_file_path = os.path.join(save_dir, f"{save_name}_nohf.md")
                    with open(md_nohf_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content_no_hf)
                        
                    result.update({
                        'md_content_path': md_file_path,
                        'md_content_nohf_path': md_nohf_file_path,
                    })
        else:
            # Simple text extraction
            image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
            origin_image.save(image_layout_path)
            
            md_content = response
            md_file_path = os.path.join(save_dir, f"{save_name}.md")
            with open(md_file_path, "w", encoding="utf-8") as md_file:
                md_file.write(md_content)
                
            result.update({
                'layout_image_path': image_layout_path,
                'md_content_path': md_file_path,
            })

        return result

    def process_directory(self, input_dir, output_dir=None, prompt_mode="prompt_layout_all_en", bbox=None, fitz_preprocess=False):
        """
        Process all images in directory structure, maintaining subdirectory structure
        For images directly in the main directory, create individual subdirectories in output
        """
        output_dir = output_dir or self.output_dir
        output_dir = os.path.abspath(output_dir)
        
        # Supported image extensions
        supported_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
        
        # Find all image files recursively
        image_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if any(file.endswith(ext) for ext in supported_extensions):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            raise ValueError(f"No supported image files found in directory {input_dir}")
        
        image_files = sorted(image_files)
        print(f"Found {len(image_files)} image files in {input_dir}")
        
        # Create tasks maintaining directory structure
        tasks = []
        for image_path in image_files:
            # Calculate relative path from input directory
            rel_path = os.path.relpath(image_path, input_dir)
            rel_dir = os.path.dirname(rel_path)
            filename = os.path.splitext(os.path.basename(rel_path))[0]
            
            # Create corresponding output directory
            if rel_dir:
                # Image is in a subdirectory - maintain the subdirectory structure
                task_output_dir = os.path.join(output_dir, rel_dir, filename)
            else:
                # Image is directly in the main directory - create individual subdirectory
                task_output_dir = os.path.join(output_dir, filename)
            
            os.makedirs(task_output_dir, exist_ok=True)
            
            tasks.append({
                'image_path': image_path,
                'prompt_mode': prompt_mode,
                'save_dir': task_output_dir,
                'save_name': filename,
                'bbox': bbox,
                'fitz_preprocess': fitz_preprocess
            })
        
        def _execute_task(task_args):
            return self._process_single_image(**task_args)
        
        # Process images with threading
        num_thread = min(len(image_files), self.num_thread)
        print(f"Processing {len(image_files)} images using {num_thread} threads...")
        
        all_results = []
        with ThreadPool(num_thread) as pool:
            with tqdm(total=len(image_files), desc="Processing images") as pbar:
                for result in pool.imap_unordered(_execute_task, tasks):
                    all_results.append(result)
                    pbar.update(1)
        
        # Save summary results
        print(f"Processing finished, results saved to {output_dir}")
        summary_file = os.path.join(output_dir, 'processing_results.jsonl')
        with open(summary_file, 'w', encoding="utf-8") as w:
            for result in all_results:
                w.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        return all_results

    def process_file_or_directory(self, input_path, output_dir=None, prompt_mode="prompt_layout_all_en", bbox=None, fitz_preprocess=False):
        """
        Process single image file or directory of images
        """
        output_dir = output_dir or self.output_dir
        output_dir = os.path.abspath(output_dir)
        
        if os.path.isdir(input_path):
            return self.process_directory(input_path, output_dir, prompt_mode, bbox, fitz_preprocess)
        elif os.path.isfile(input_path):
            # Single image file - create individual subdirectory
            filename = os.path.splitext(os.path.basename(input_path))[0]
            save_dir = os.path.join(output_dir, filename)
            os.makedirs(save_dir, exist_ok=True)
            
            result = self._process_single_image(input_path, prompt_mode, save_dir, filename, bbox, fitz_preprocess)
            
            # Save result summary in the main output directory
            with open(os.path.join(output_dir, 'processing_results.jsonl'), 'w', encoding="utf-8") as w:
                w.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            return [result]
        else:
            raise ValueError(f"Input path {input_path} does not exist")


def main():
    prompts = list(dict_promptmode_to_prompt.keys())
    parser = argparse.ArgumentParser(
        description="Convert image directories to OCR results (markdown/json)"
    )
    
    parser.add_argument(
        "input_path", type=str,
        help="Input image file or directory containing images"
    )
    
    parser.add_argument(
        "--output", type=str, default="./ocr_output",
        help="Output directory (default: ./ocr_output)"
    )
    
    parser.add_argument(
        "--prompt", choices=prompts, type=str, default="prompt_layout_all_en",
        help="Prompt mode for OCR processing"
    )
    
    parser.add_argument(
        '--bbox', 
        type=int, 
        nargs=4, 
        metavar=('x1', 'y1', 'x2', 'y2'),
        help='Bounding box for grounding OCR (required for prompt_grounding_ocr)'
    )
    
    parser.add_argument(
        "--ip", type=str, default="localhost",
        help="VLLM server IP (default: localhost)"
    )
    
    parser.add_argument(
        "--port", type=int, default=8000,
        help="VLLM server port (default: 8000)"
    )
    
    parser.add_argument(
        "--model_name", type=str, default="model",
        help="Model name for VLLM server (default: model)"
    )
    
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="Sampling temperature (default: 0.1)"
    )
    
    parser.add_argument(
        "--top_p", type=float, default=1.0,
        help="Top-p sampling parameter (default: 1.0)"
    )
    
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="DPI for image processing (default: 200)"
    )
    
    parser.add_argument(
        "--max_completion_tokens", type=int, default=16384,
        help="Maximum completion tokens (default: 16384)"
    )
    
    parser.add_argument(
        "--num_thread", type=int, default=32,
        help="Number of threads for parallel processing (default: 32)"
    )
    
    parser.add_argument(
        "--no_fitz_preprocess", action='store_true',
        help="Disable fitz preprocessing (default: False)"
    )
    
    parser.add_argument(
        "--min_pixels", type=int, default=None,
        help="Minimum pixels for image resizing"
    )
    
    parser.add_argument(
        "--max_pixels", type=int, default=None,
        help="Maximum pixels for image resizing"
    )
    
    args = parser.parse_args()

    converter = ImagesToOCRConverter(
        ip=args.ip,
        port=args.port,
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_tokens=args.max_completion_tokens,
        num_thread=args.num_thread,
        dpi=args.dpi,
        output_dir=args.output,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    fitz_preprocess = not args.no_fitz_preprocess
    if fitz_preprocess:
        print("Using fitz preprocess for image input")
    
    results = converter.process_file_or_directory(
        args.input_path, 
        prompt_mode=args.prompt,
        bbox=args.bbox,
        fitz_preprocess=fitz_preprocess,
    )
    
    print(f"Successfully processed {len(results)} images")


if __name__ == "__main__":
    main()
