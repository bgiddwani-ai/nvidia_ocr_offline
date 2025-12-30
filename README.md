
## Deploy Server

docker run --gpus device=0 -it --ipc=host -p 8000:8000 "container"


## Installation

pip install -e .

pip install -r requirements.txt



## Client Inference 

A. PDFs to Images (Optional- If pdfs are present)

Directory -> .PDFs

python3 ocr/pdf_to_imgs.py /path/to/inputdir --output /path/to/outputdir

B. Images to OCR 


python3 ocr/ocr.py /path/to/inputdir --output /path/to/outputdir --port 8000 --num_thread



