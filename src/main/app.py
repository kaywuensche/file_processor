from typing import List
from fastapi import FastAPI, File, UploadFile, Form, Response, Path
from utils import save_as_temp, convert_office_to_pdf, pdf_2_pages, augmentation
from fpdf import FPDF
import datetime
import fitz
import camelot
import subprocess
import re
from PIL import Image, ImageOps, ImageDraw
import random
import zipfile
import pandas as pd
import enchant
from langdetect import detect_langs
from spellchecker import SpellChecker
from translate import Translator
from spacy import displacy
from pathlib import Path
import tempfile
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import os
import spacy
import cv2
import io
import shutil
from pdf2image import convert_from_bytes
from PyPDF2 import PdfFileWriter, PdfFileReader, PdfFileMerger
import moviepy.editor as mp

app = FastAPI(title='File processing',
              description='<b> API for file processing. This includes file convertion as well as text and image extraction and preparation.<br><br><br>Contact the developer:<br><font color="#808080">Kay Wünsche: <a href="mailto:">kay.wuensche@gmx.de</a>')

#load enchent dictonaries
dic_en = enchant.Dict("en_US")
dic_de = enchant.Dict("de_DE")

#load default spacy models
nlp_de_default = spacy.load("de_dep_news_trf")
nlp_en_default = spacy.load("en_core_web_trf")

allowed_office_file = ['.docx','.pptx']
allowed_image_files = ['.png', '.jpeg', '.jpg']

@app.post("/excel_to_sheet_names", tags=['services for excel'])
async def excel_to_sheet_names(
    xlsx_file: bytes = File(..., description="Upload a .xlsx file for sheetname extraction:")
):
    """This endpoint takes a .xlsx file and returns the sheetnames."""
    xl = pd.ExcelFile(io.BytesIO(xlsx_file))
    return {"sheetnames": xl.sheet_names}

@app.post("/excel_to_sheet_text", tags=['services for excel'])
async def excel_to_sheet_text(
    xlsx_file: bytes = File(..., description="Upload a .xlsx file:"),
    sheet_name: str = Form(..., description="Sheet name for text extraction:")
):
    """This endpoint takes a .xlsx file and a sheet name and returns the text of the sheet."""
    xl = pd.ExcelFile(io.BytesIO(xlsx_file))
    text = xl.parse(sheet_name)
    text = text.fillna('')
    return text.to_dict()

@app.post("/office_doc_to_pdf", tags=['services for docx and pptx'])
async def office_doc_to_pdf(
    office_file: UploadFile = File(..., description="Upload a .docx or .pptx file for pdf convertion:")
):
    """This endpoint takes a .docx or .pptx file and returns the pdf version."""
    temp_file, extension = save_as_temp(office_file)
    if extension in allowed_office_file:
        pdf_file = convert_office_to_pdf(os.path.dirname(temp_file), temp_file)
        with open(pdf_file, 'rb') as file_data:
            binary_file = file_data.read()
        os.remove(temp_file)
        os.remove(pdf_file)
        return Response(content=binary_file, media_type="application/pdf")
    else:
        return "not a docx or pptx file"

@app.post("/office_doc_to_embedded_images", tags=['services for docx and pptx'])
async def office_doc_to_embedded_images(
    office_file: UploadFile = File(..., description="Upload a .docx or .pptx file for embedded image extraction:")
):
    """This endpoint takes a .docx or .pptx file and returns the embedded images."""
    temp_path, extension = save_as_temp(office_file)
    if extension in allowed_office_file:
        if extension.replace(".", "") == "docx":
            extension_name = "word"
        else:
            extension_name = "ppt"
    else:
        return "not a docx or pptx file"
    # rename to uploded file to zip file
    zipfile_media = temp_path.replace(extension, '.zip')
    os.rename(temp_path, zipfile_media)
    # unzip file to get media folder
    try:
        with tempfile.TemporaryDirectory() as target_temp:
            # unzip file to get images from media folder
            zip_ref = zipfile.ZipFile(zipfile_media, 'r')
            zip_ref.extractall(target_temp)
            zip_ref.close()
            # create a ZipFile object for returning the images
            zip_file = os.path.join(target_temp + 'embedded_images.zip')
            shutil.make_archive(os.path.join(target_temp + 'embedded_images'), 'zip', os.path.join(target_temp, extension_name, "media", ""))
            #zip to binary for return
            with open(zip_file, 'rb') as file_data:
                binary_file = file_data.read()
            os.remove(zip_file)
            os.remove(zipfile_media)
            return Response(content=binary_file , media_type="application/x-zip-compressed")
    except:
        os.remove(zipfile_media)
        shutil.rmtree(target_temp)
        return "some error occur by unziping"

@app.post("/create_pdf", tags=['services for pdf'])
async def create_pdf(
    text: str = Form(None, description="Text as content for the pdf:"),
    image_file: UploadFile = File(None, description="Image as content for the pdf:")
):
    """This endpoint takes a text and and a image and returns a pdf page."""
    #generate empty pdf
    pdf = FPDF()
    pdf.add_page()
    #format pdf
    pdf.set_font('Arial', 'B')
    pdf.ln(10)
    pdf.cell(180, 8, txt='File processing', ln=1, align="R")
    pdf.set_font('Arial')
    pdf.cell(180, 8, txt='Test document', ln=1, align="R")
    date = str(datetime.datetime.now().strftime("%d/%m/%Y %H:%M"))
    pdf.cell(180, 8, txt=date, ln=1, align="R")
    pdf.cell(180, 8, txt='Page x', ln=1, align="R")
    pdf.ln(10)
    # add text
    if text is not None:
        pdf.cell(20, 10, txt='Test content: ' + text, ln=1, align="L")
        pdf.ln(5)
    #add image
    if image_file is not None:
        image_path, extension = save_as_temp(image_file)
        pdf.image(image_path, link =None)
        pdf.cell(20, 10, txt="{}".format(image_path), ln=1)
        os.remove(image_path)
    _, path = tempfile.mkstemp(prefix='protocol', suffix='.pdf')
    pdf.output(path)
    os.close(_)
    with open(path, 'rb') as file_data:
        binary_file = file_data.read()
    os.remove(path)
    return Response(content=binary_file, media_type="application/pdf")

@app.post("/split_pdf_to_pages", tags=['services for pdf'])
async def split_pdf_to_pages(
    file: bytes = File(..., description="Upload a pdf for splitting in pages:")
):
    """This endpoint takes a pdf file and returns single pdf pages as a .zip file."""
    with tempfile.TemporaryDirectory() as session:
        #split pdf to pages
        pdf_2_pages(session, file)
        # create a ZipFile object for returning the pdf pages
        zip_file = os.path.join(os.path.dirname(session), 'pdf_pages.zip')
        shutil.make_archive(os.path.splitext(zip_file)[0], 'zip', session)
        # zip to binary for return
        with open(zip_file, 'rb') as file_data:
            binary_file = file_data.read()
        os.remove(zip_file)
    return Response(content=binary_file, media_type="application/x-zip-compressed")

@app.post("/pdf_page_to_image", tags=['services for pdf'])
async def pdf_page_to_image(
    pdf_file: bytes = File(..., description="Upload a pdf page:")
):
    """This endpoint takes pdf page and returns a jpeg file of the page."""
    pages = convert_from_bytes(pdf_file, 1000)
    byteIO = io.BytesIO()
    pages[0].save(byteIO, 'JPEG')
    return Response(content=byteIO.getvalue(), media_type="image/jpeg")

@app.post("/pdf_to_embedded_images", tags=['services for pdf'])
async def pdf_to_embedded_images(
    pdf_file: bytes = File(..., description="Upload a pdf file:")
):
    """This endpoint takes a pdf file and returns the embedded images."""
    with tempfile.TemporaryDirectory() as target_temp:
        doc = fitz.open(stream=io.BytesIO(pdf_file), filetype="pdf")
        for i in range(len(doc)):
            for img in doc.getPageImageList(i):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n < 5:  # this is GRAY or RGB
                    pix.writePNG(os.path.join(target_temp, "p%s-%s.jpeg") % (i, xref))
                else:  # CMYK: convert to RGB first
                    pix1 = fitz.Pixmap(fitz.csRGB, pix)
                    pix1.writePNG(os.path.join(target_temp, "p%s-%s.jpeg") % (i, xref))
        # create a ZipFile object for returning the images
        zip_file = os.path.join(os.path.dirname(target_temp), 'images.zip')
        shutil.make_archive(os.path.splitext(zip_file)[0], 'zip', target_temp)
        # zip to binary for return
        with open(zip_file, 'rb') as file_data:
            binary_file = file_data.read()
        os.remove(zip_file)
    return Response(content=binary_file, media_type="application/x-zip-compressed")

@app.post("/pdf_to_text", tags=['services for pdf'])
async def pdf_to_text(
    pdf_file: bytes = File(..., description="Upload a pdf file:")
):
    """This endpoint takes a pdf and returns the extracted text page by page."""
    doc = fitz.open(stream=io.BytesIO(pdf_file), filetype="pdf")
    result = dict()
    for i in range(0, doc.pageCount):
        page = doc.loadPage(i)
        pagetext = page.getText("text").encode("utf-8").decode("utf-8")
        result['page_' + str(i)] = [text.strip() for text in pagetext.split('\n') if text.strip() != '']
    return result

@app.post("/pdf_page_to_text_blocks", tags=['services for pdf'])
async def pdf_page_to_text_blocks(
    pdf_file: UploadFile = File(..., description="Upload a pdf page:")
):
    """This endpoint takes a pdf and returns the extracted text block by block as list."""
    temp_path_pdf, extension = save_as_temp(pdf_file)
    command = 'pdftotext -layout -enc UTF-8 ' + temp_path_pdf
    cmd = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd='/', shell=True)
    cmd.communicate()
    temp_path_txt= temp_path_pdf.replace('.pdf', '.txt')
    result = {}
    with open(temp_path_txt, "r") as txt:
        texte = txt.read()
        result['text_blocks'] = [re.sub(r"[\s^;]{2,}", "", text.replace('\n', ' ').strip()) for text in texte.split('\n\n') if text.strip() != '']
    os.remove(temp_path_pdf)
    os.remove(temp_path_txt)
    return result

@app.post("/pdf_page_to_table", tags=['services for pdf'])
async def pdf_to_table(
    pdf_file: UploadFile = File(..., description="Upload a pdf file for table extraction:")
):
    """This endpoint takes a pdf and returns the content of tables."""
    temp_path, extension = save_as_temp(pdf_file)
    tables = camelot.read_pdf(temp_path , pages="all")
    result = {}
    for i in range(0, tables.n):
        result['table_' + str(i)]= tables[i].df
    os.remove(temp_path)
    return result

@app.post("/pdf_page_highlighting", tags=['services for pdf'])
async def pdf_page_highlighting(
    pdf_file: bytes = File(..., description="Upload a pdf page:"),
    text_for_matching: List[str] = Form(..., description="Text for highlighting:")
):
    """This endpoint takes a pdf and a list of strings. As the result it returns the pdf including text highlighting."""
    matching_list = text_for_matching[0].split(',')
    doc = fitz.open(stream=io.BytesIO(pdf_file), filetype="pdf")
    for page in doc:
        for i in range(0, len(matching_list)):
            text = str(matching_list[i]).replace("_", " ")
            ql = page.searchFor(text)
            page.addHighlightAnnot(ql)
    return Response(content=doc.tobytes(), media_type="application/pdf")

@app.post("/merge_pdf", tags=['services for pdf'])
async def merge_pdf(
    pdf_files: List[bytes] = File(..., description="Upload multiple pdf files:")
):
    """This endpoint takes multiple pdf files and merged them into one pdf file."""
    merger = PdfFileMerger()
    for pdf in pdf_files:
        pdf = PdfFileReader(io.BytesIO(pdf))
        merger.append(pdf, 'rb')
    byteIO = io.BytesIO()
    merger.write(byteIO)
    merger.close()
    return Response(content=byteIO.getvalue(), media_type="application/pdf")

@app.post("/video_to_frames", tags=['services for video'])
async def video_to_frames(
    video_file: UploadFile = File(..., description="Upload a video for frame extraction:"),
    frame_rate: int = Form(None, description="Frame will be extracted every x seconds:")):
    """This endpoint takes a video and returns a zip file with extracted video frames"""
    if frame_rate is None:
        frame_rate = 1
    #open video
    path, extension = save_as_temp(video_file)
    vidcap = cv2.VideoCapture(path)
    count = 0
    second = 0
    success = True
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    # get frames
    with tempfile.TemporaryDirectory() as session:
        while success:
            success, image = vidcap.read()
            if success and count % (frame_rate * fps) == 0:
                second += 1
                frame = os.path.join(session, 'second_%d.jpg' % second)
                cv2.imwrite(frame, image)
            count += 1
        # create a ZipFile object for returning the images
        zip_file = os.path.join(os.path.dirname(session), 'frames.zip')
        shutil.make_archive(os.path.splitext(zip_file)[0], 'zip', session)
    # zip to binary for return
    vidcap.release()
    with open(zip_file, 'rb') as file_data:
        binary_file = file_data.read()
    os.remove(zip_file)
    os.remove(path)
    return Response(content=binary_file, media_type="application/x-zip-compressed")

@app.post('/video_to_audio', tags=['services for video'])
async def video_to_audio(
    video_file: UploadFile = File(..., description="Video file for audio extraction:")
):
    """This endpoint takes a video and returns the extracted audio file."""
    temp_path, extension = save_as_temp(video_file)
    videoclip = mp.VideoFileClip(temp_path)
    audioclip = videoclip.audio
    _, path = tempfile.mkstemp(prefix='audio_', suffix='.mp3')
    audioclip.write_audiofile(path)
    with open(path, 'rb') as file_data:
        binary_file = file_data.read()
    os.remove(temp_path)
    os.remove(path)
    return Response(content=binary_file, media_type="application/mp3")

@app.post('/image_to_jpeg', tags=['services for images'])
async def img_to_jpeg(
    image_file: bytes = File(..., description="Image file for jpeg convertion:")
):
    """This endpoint takes an image and returns the jpg version."""
    with Image.open(io.BytesIO(image_file)).convert('RGB') as img:
        byteIO = io.BytesIO()
        img.save(byteIO, 'JPEG')
    return Response(content=byteIO.getvalue(), media_type="image/jpg")

@app.post('/resize_image', tags=['services for images'])
async def resize_image(
    image_file: bytes = File(..., description="Image file for resizing:"),
    basewidth: int = Form(..., description="Basewidth for resizing:")):
    """This endpoint takes an image and returns a resized version."""
    with Image.open(io.BytesIO(image_file)).convert('RGB') as img:
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        byteIO = io.BytesIO()
        img.save(byteIO, 'JPEG')
    return Response(content=byteIO.getvalue(), media_type="image/jpg")

@app.post('/grayscale_image', tags=['services for images'])
async def grayscale_image(
    image_file: bytes = File(..., description="Image file for conversion to a black and white image:")
):
    """This endpoint takes an image and returns the grayscale version of the image."""
    with Image.open(io.BytesIO(image_file)).convert('RGB') as img:
        gray_image = ImageOps.grayscale(img)
        byteIO = io.BytesIO()
        gray_image.save(byteIO, 'JPEG')
    return Response(content=byteIO.getvalue(), media_type="image/jpg")

@app.post('/image_augmentation', tags=['services for images'])
async def image_augmentation(
    image_file: UploadFile = File(..., description="Image file for augmentation")
):
    """This endpoint takes an image and returns an augmented version."""
    path, extension = save_as_temp(image_file)
    path = augmentation(path, random.randint(0, 40))
    with open(path, 'rb') as file_data:
        binary_file = file_data.read()
    os.remove(path)
    return Response(content=binary_file, media_type="image/" + extension.replace('.', ''))

@app.post('/draw_bbox_on_image', tags=['services for images'])
async def draw_bbox_on_image(
    image_file: bytes = File(..., description="Image file for drawing"),
    bbox: str = Form(..., description="Boundingbox in the format 'x_min y_min x_max y_max': ")
):
    """This endpoint takes an image and a bounding box and returns a tagged version."""
    with Image.open(io.BytesIO(image_file)).convert('RGB') as img:
        draw = ImageDraw.Draw(img)
        bbox = bbox.split()
        draw.rectangle(((float(bbox[0]), float(bbox[1])), (float(bbox[2]), float(bbox[3]))), fill=None, outline='red')
        byteIO = io.BytesIO()
        img.save(byteIO, 'JPEG')
        return Response(content=byteIO.getvalue(), media_type="image/jpg")

@app.post('/crop_image', tags=['services for images'])
async def crop_image(
    image_file: bytes = File(..., description="Image file for croping"),
    bbox: str = Form(..., description="Boundingbox in the format 'x_min y_min x_max y_max': ")
):
    """This endpoint takes an image and a bounding box and returns a croped version."""
    with Image.open(io.BytesIO(image_file)).convert('RGB') as img:
        bbox = bbox.split()
        img_cropped = img.crop((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))
        byteIO = io.BytesIO()
        img_cropped.save(byteIO, 'JPEG')
        return Response(content=byteIO.getvalue(), media_type="image/jpg")

@app.post('/language_classification', tags=['services for text'])
async def language_classification(
    text: str = Form(..., description="Text for language classification:")
):
    """This endpoint takes text and returns the corresponding language. Text will be classified to english or german only."""
    method = language = confidence = "NA"
    try:
        #get language by model
        language_list = detect_langs(text)
        for i in range(0, len(language_list)):
            lang_abr, conf = str(language_list[i]).split(':')
            if (lang_abr == "en" or lang_abr == "de") and float(conf) > 0.5:
                language = lang_abr
                confidence = conf
                method = "prediction"
                break
        # get language by dictionaries
        if language == "NA":
            en_count = de_count = 0
            words = text.split(" ")
            for word in words:
                if dic_en.check(word):
                    en_count += 1
                if dic_de.check(word):
                    de_count += 1
            if en_count > 0 or de_count > 0:
                method = "lookup"
                if de_count >= en_count:
                    language = "de"
                else:
                    language = "en"
            else:
                method = "default"
                language = "de"
    except:
        method = "default"
        language = "de"
    result = dict()
    result['text'] = text
    result['language'] = language
    result['method'] = method
    result['confidence'] = confidence
    return result

@app.post('/grammatical_analysis', tags=['services for text'])
async def grammatical_analysis(
    text: str = Form(..., description="Text for grammatical analysis:"),
    language: str = Form(..., description="Language of text (en or de):")
):
    """This endpoint takes text and returns part of speech and their relations."""
    if (language == "en"):
        doc = nlp_en_default(text)
    elif (language == "de"):
        doc = nlp_de_default(text)
    else:
        return 'language model not found'
    #get token
    result = list()
    for token in doc:
        token_list = dict()
        token_list["token"] = token.text
        token_list["position"] = token.i
        token_list["lemma"] = token.lemma_
        token_list["part_of_speech"] = token.pos_
        token_list["tag"] = token.tag_
        token_list["dep"] = token.dep_
        if (token.head.text != ""):
            token_list["head_text"] = token.head.text
            token_list["head_part_of_speech"] = token.head.pos_
        token_list["has_characters"] = token.is_alpha
        token_list["is_digit"] = token.is_digit
        token_list["is_number"] = token.like_num
        token_list["is_punctation"] = token.is_punct
        token_list["is_bracket"] = token.is_bracket
        token_list["is_quote"] = token.is_quote
        token_list["is_mail"] = token.like_email
        token_list["is_currency"] = token.is_currency
        token_list["is_sentiment"] = token.sentiment
        token_list["is_stopword"] = token.is_stop
        child_list = [child for child in token.children]
        if (len(child_list) > 0):
            for i in range(0, len(child_list)):
                if "children" in token_list:
                    token_list["children"].append(str(child_list[i]))
                else:
                    token_list["children"] = [str(child_list[i])]
        result.append(token_list)
    return result

@app.post('/grammatical_analysis_on_image', tags=['services for text'])
async def grammatical_analysis_on_image(
    text: str = Form(..., description="Text for grammatical analysis:"),
    language: str = Form(..., description="Language of text (en or de):")
):
    """This endpoint takes text and returns part of speech and their relations on an image."""
    if (language == "en"):
        doc = nlp_en_default(text)
    elif (language == "de"):
        doc = nlp_de_default(text)
    else:
        return 'language model not found'
    #get svg
    svg_file = displacy.render(doc, style="dep", jupyter=False)
    #save as temp file
    _, path_svg = tempfile.mkstemp(prefix='upload_', suffix='.svg')
    path_png = path_svg.replace('.svg','.png')
    output_path = Path(path_svg)
    output_path.open("w", encoding="utf-8").write(svg_file)
    #convert to png
    drawing = svg2rlg(path_svg)
    renderPM.drawToFile(drawing, path_png, fmt="PNG")
    #convert to binary
    with open(path_png, 'rb') as file_data:
        png_file = file_data.read()
    os.remove(path_svg)
    os.remove(path_png)
    return Response(content=png_file, media_type="image/png")

@app.post('/get_chunks', tags=['services for text'])
async def get_chunks(
    text: str = Form(..., description="Text for chunks / phrases detection:"),
    language: str = Form(..., description="Language of text (en or de):")
):
    """This endpoint takes text and returns chunks / phrases."""
    if (language == "en"):
        doc = nlp_en_default(text)
    elif (language == "de"):
        doc = nlp_de_default(text)
    else:
        return 'language model not found'
    result = list()
    for chunk in doc.noun_chunks:
        chunk_list = dict()
        chunk_list["chunk"] = chunk.text
        chunk_list["root_chunk"] = chunk.root.text
        chunk_list["chunk_dep"] = chunk.root.dep_
        chunk_list["chunk_head"] = chunk.root.head.text
        result.append(chunk_list)
    return result

@app.post('/spell_checking', tags=['services for text'])
async def spell_checking(
    text: str = Form(..., description="Text for spell checking:"),
    language: str = Form(..., description="Language of text (en or de):")
):
    """This endpoint takes text and returns the correction by spell checking."""
    text = text.strip()
    result = dict()
    result['text'] = text
    result['language'] = language
    if (language == "en"):
        spell = SpellChecker(language='en')
    elif (language == "de"):
        spell = SpellChecker(language='de')
    else:
        return 'language model not found'
    # find those words that may be misspelled
    misspelled = spell.unknown(text.split(" "))
    for word in misspelled:
        text = text.replace(word, spell.correction(word))
    if(result['text'] != text):
        result['correction'] = text
    else:
        result['correction'] = "NA"
    return result

@app.post('/translation', tags=['services for text'])
async def translation(
    text_to_translate: str = Form(..., description="Text for translation:"),
    language: str = Form(..., description="Language of text (en or de):")
):
    """This endpoint takes text and returns the english or german translation using google translation api."""
    if(language == "de"):
        german_query = text_to_translate.strip()
        translator = Translator(from_lang="german", to_lang="english")
        english_query = translator.translate(text_to_translate)
    elif(language == "en"):
        english_query = text_to_translate.strip()
        translator = Translator(from_lang="english", to_lang="german")
        german_query = translator.translate(text_to_translate)
    else:
        return 'language model not found'
    result = {'german_query': german_query, 'english_query': english_query}
    return result