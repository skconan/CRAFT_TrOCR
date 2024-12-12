import uvicorn
import aiofiles
import json
from fastapi import FastAPI, UploadFile, File, HTTPException

from encrypt import encrypt_data
from ocr import TextRecognizer
from utils.file import read_media

STORAGE_DIR = "./storage/media"

app = FastAPI()


ocr_weights = "./weights/trocr-base-handwritten/"
# ocr_weights = "./weights/thai-trocr/"
craft_weight = "./weights/craft_mlt_25k.pth"
refine_weight = "./weights/craft_refiner_CTW1500.pth"

recognizer = TextRecognizer(craft_weight, refine_weight, ocr_weights)


@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...),
):
    contents = await file.read()
    file_path = f"{STORAGE_DIR}/{file.filename}"
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(contents)

    img_list = read_media(file_path)
    try:
        text_results = recognizer.recognize(img_list)
        return {"results": encrypt_data(json.dumps(text_results))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
