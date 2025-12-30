import os
import glob
import argparse
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

from nv_ocr.utils.doc_utils import load_images_from_pdf


class PDFToImagesConverter:
    """
    Convert PDF files to image directories
    """
    
    def __init__(self, dpi=200, num_thread=32, output_dir="./pdf_images"):
        self.dpi = dpi
        self.num_thread = num_thread
        self.output_dir = output_dir
    
    def convert_single_pdf(self, pdf_path, save_dir):
        """
        Convert a single PDF to images
        """
        print(f"Loading PDF: {pdf_path}")
        images = load_images_from_pdf(pdf_path, dpi=self.dpi)
        
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        pdf_output_dir = os.path.join(save_dir, pdf_name)
        os.makedirs(pdf_output_dir, exist_ok=True)
        
        results = []
        for i, image in enumerate(images):
            page_num = i + 1  # Start from 1
            image_name = f"{pdf_name}_page{page_num}.png"
            image_path = os.path.join(pdf_output_dir, image_name)
            
            # Save image
            image.save(image_path, "PNG")
            
            results.append({
                'pdf_path': pdf_path,
                'page_number': page_num,
                'image_path': image_path,
                'image_name': image_name
            })
        
        print(f"Converted {len(images)} pages from {pdf_name}")
        return results
    
    def convert_directory(self, input_dir, output_dir=None):
        """
        Convert all PDF files in a directory
        """
        output_dir = output_dir or self.output_dir
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all PDF files
        pdf_files = []
        for pattern in ["*.pdf", "*.PDF"]:
            pdf_files.extend(glob.glob(os.path.join(input_dir, pattern)))
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in directory {input_dir}")
        
        pdf_files = sorted(pdf_files)
        print(f"Found {len(pdf_files)} PDF files in {input_dir}")
        
        # Create tasks
        tasks = [(pdf_path, output_dir) for pdf_path in pdf_files]
        
        def _execute_task(task_args):
            pdf_path, save_dir = task_args
            return self.convert_single_pdf(pdf_path, save_dir)
        
        # Process PDFs with threading
        num_thread = min(len(pdf_files), self.num_thread)
        print(f"Converting {len(pdf_files)} PDFs using {num_thread} threads...")
        
        all_results = []
        with ThreadPool(num_thread) as pool:
            with tqdm(total=len(pdf_files), desc="Converting PDFs") as pbar:
                for results in pool.imap_unordered(_execute_task, tasks):
                    all_results.extend(results)
                    pbar.update(1)
        
        print(f"Conversion finished. Images saved to {output_dir}")
        return all_results
    
    def convert_file(self, input_path, output_dir=None):
        """
        Convert PDF file(s) to images
        """
        output_dir = output_dir or self.output_dir
        output_dir = os.path.abspath(output_dir)
        
        if os.path.isdir(input_path):
            return self.convert_directory(input_path, output_dir)
        elif input_path.lower().endswith(('.pdf', '.PDF')):
            os.makedirs(output_dir, exist_ok=True)
            return self.convert_single_pdf(input_path, output_dir)
        else:
            raise ValueError(f"Input must be a PDF file or directory containing PDFs")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF files to image directories"
    )
    
    parser.add_argument(
        "input_path", type=str,
        help="Input PDF file or directory containing PDF files"
    )
    
    parser.add_argument(
        "--output", type=str, default="./pdf_images",
        help="Output directory for images (default: ./pdf_images)"
    )
    
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="DPI for image conversion (default: 200)"
    )
    
    parser.add_argument(
        "--num_thread", type=int, default=32,
        help="Number of threads for parallel processing (default: 32)"
    )
    
    args = parser.parse_args()
    
    converter = PDFToImagesConverter(
        dpi=args.dpi,
        num_thread=args.num_thread,
        output_dir=args.output
    )
    
    results = converter.convert_file(args.input_path, args.output)
    print(f"Successfully converted {len(results)} pages to images")


if __name__ == "__main__":
    main()