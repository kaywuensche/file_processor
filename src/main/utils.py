import subprocess
import re
import sys
from PyPDF2 import PdfFileWriter, PdfFileReader, PdfFileMerger
import tempfile
import os
from fastapi import File, UploadFile
from imgaug import augmenters as iaa
import cv2
import io

def save_as_temp(file: UploadFile = File(...)):
    extension = os.path.splitext(file.filename)[1]
    # make a temp file
    _, path = tempfile.mkstemp(prefix='upload_', suffix=extension)
    # write file in chunks to not explode in RAM usage
    with open(path, 'ab') as f:
        for chunk in iter(lambda: file.file.read(10000), b''):
            f.write(chunk)
    os.close(_)
    return path, extension

# libre office method for handling pptx and docx
def run(*popenargs, **kwargs):
    input = kwargs.pop("input", None)
    check = kwargs.pop("handle", False)
    if input is not None:
        if 'stdin' in kwargs:
            raise ValueError('stdin and input arguments may not both be used.')
        kwargs['stdin'] = subprocess.PIPE
    process = subprocess.Popen(*popenargs, **kwargs)
    try:
        stdout, stderr = process.communicate(input)
    except:
        process.kill()
        process.wait()
        raise
    retcode = process.poll()
    if check and retcode:
        raise subprocess.CalledProcessError(
            retcode, process.args, output=stdout, stderr=stderr)
    return retcode, stdout, stderr

def convert_office_to_pdf(folder, source, timeout=None):
    args = [libreoffice_exec(), '--headless', '--convert-to', 'pdf', '--outdir', folder, source]
    process = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    filename = re.search('-> (.*?) using filter', process.stdout.decode())
    if filename is None:
        raise LibreOfficeError(process.stdout.decode())
    else:
        return filename.group(1)

def libreoffice_exec():
    # TODO: Provide support for more platforms
    if sys.platform == 'darwin':
        return '/Applications/LibreOffice.app/Contents/MacOS/soffice'
    return 'libreoffice'

class LibreOfficeError(Exception):
    def __init__(self, output):
        self.output = output

def pdf_2_pages(pdf_pages, file):
    inputpdf = PdfFileReader(io.BytesIO(file))
    for i in range(0, inputpdf.getNumPages()):
        output = PdfFileWriter()
        output.addPage(inputpdf.getPage(i))
        with open(os.path.join(pdf_pages, "page" + str(i) + ".pdf"), "wb") as outputStream:
            output.write(outputStream)

def augmentation(img_path, aug):
        sequences = list()
        if (aug == 0):
            sequences.append(("aug_1", iaa.Sequential(
                [iaa.GaussianBlur(sigma=(0.0, 2.0)), iaa.ElasticTransformation(alpha=(0, 1.1), sigma=0.05)])))
        elif (aug == 1):
            sequences.append(("aug_2", iaa.Sequential([iaa.EdgeDetect(0.02), iaa.Dropout(0.12)])))
        elif (aug == 2):
            sequences.append(("aug_3", iaa.Sequential([iaa.AdditiveGaussianNoise(scale=0.1 * 255)])))
        elif (aug == 3):
            sequences.append(("aug_4", iaa.Sequential(
                [iaa.AdditiveGaussianNoise(scale=0.1 * 255), iaa.Sharpen(alpha=(0, 0.25), lightness=(0.5, 0.8))])))
        elif (aug == 4):
            sequences.append(("aug_5", iaa.Sequential([iaa.ContrastNormalization((1.5, 2.5))])))
        elif (aug == 5):
            sequences.append(("aug_6", iaa.Sequential([iaa.Add(0, per_channel=True), iaa.Salt(p=0.05)])))
        elif (aug == 6):
            sequences.append(("aug_7", iaa.Sequential([iaa.Add(0, per_channel=True), iaa.Pepper(p=0.05)])))
        elif (aug == 7):
            sequences.append(("aug_8", iaa.Sequential([iaa.Add(5, per_channel=True), iaa.Salt(p=0.05)])))
        elif (aug == 8):
            sequences.append(("aug_9", iaa.Sequential([iaa.Add(5, per_channel=True), iaa.Pepper(p=0.05)])))
        elif (aug == 9):
            sequences.append(("aug_10", iaa.Sequential([iaa.Add(5, per_channel=True), iaa.ContrastNormalization(2.0)])))
        elif (aug == 10):
            sequences.append(
                ("aug_11", iaa.Sequential([iaa.Add(5, per_channel=True), iaa.ContrastNormalization(1.75)])))
        elif (aug == 11):
            sequences.append(("aug_12", iaa.Sequential([iaa.Add(5, per_channel=True), iaa.ContrastNormalization(1.5)])))
        elif (aug == 12):
            sequences.append(
                ("aug_13", iaa.Sequential([iaa.Add(5, per_channel=True), iaa.ContrastNormalization(1.25)])))
        elif (aug == 13):
            sequences.append(("aug_14", iaa.Sequential([iaa.Add(5, per_channel=True), iaa.ContrastNormalization(1.0)])))
        elif (aug == 14):
            sequences.append(("aug_15", iaa.Sequential([iaa.Add(0, per_channel=True), iaa.ContrastNormalization(2.0)])))
        elif (aug == 15):
            sequences.append(
                ("aug_16", iaa.Sequential([iaa.Add(0, per_channel=True), iaa.ContrastNormalization(1.75)])))
        elif (aug == 16):
            sequences.append(("aug_17", iaa.Sequential([iaa.Add(0, per_channel=True), iaa.ContrastNormalization(1.5)])))
        elif (aug == 17):
            sequences.append(
                ("aug_18", iaa.Sequential([iaa.Add(0, per_channel=True), iaa.ContrastNormalization(1.25)])))
        elif (aug == 18):
            sequences.append(("aug_19", iaa.Sequential([iaa.Add(0, per_channel=True), iaa.ContrastNormalization(1.0)])))
        elif (aug == 19):
            sequences.append(
                ("aug_20", iaa.Sequential([iaa.Add(-5, per_channel=True), iaa.ContrastNormalization(2.0)])))
        elif (aug == 20):
            sequences.append(
                ("aug_21", iaa.Sequential([iaa.Add(-5, per_channel=True), iaa.ContrastNormalization(1.75)])))
        elif (aug == 21):
            sequences.append(
                ("aug_22", iaa.Sequential([iaa.Add(-5, per_channel=True), iaa.ContrastNormalization(1.5)])))
        elif (aug == 22):
            sequences.append(
                ("aug_23", iaa.Sequential([iaa.Add(-5, per_channel=True), iaa.ContrastNormalization(1.25)])))
        elif (aug == 23):
            sequences.append(
                ("aug_24", iaa.Sequential([iaa.Add(-5, per_channel=True), iaa.ContrastNormalization(1.0)])))
        elif (aug == 24):
            sequences.append(("aug_25", iaa.Sequential([iaa.Add(0, per_channel=True), iaa.Emboss(alpha=0.5)])))
        elif (aug == 25):
            sequences.append(("aug_26", iaa.Sequential([iaa.Add(0, per_channel=True), iaa.Emboss(alpha=0.75)])))
        elif (aug == 26):
            sequences.append(("aug_27", iaa.Sequential([iaa.Add(0, per_channel=True), iaa.Emboss(alpha=1.0)])))
        elif (aug == 27):
            sequences.append(("aug_28", iaa.Sequential([iaa.Add(5, per_channel=True), iaa.Emboss(alpha=0.5)])))
        elif (aug == 28):
            sequences.append(("aug_29", iaa.Sequential([iaa.Add(5, per_channel=True), iaa.Emboss(alpha=1.0)])))
        elif (aug == 29):
            sequences.append(("aug_30", iaa.Sequential([iaa.Add(5, per_channel=True), iaa.Emboss(alpha=0.75)])))
        elif (aug == 30):
            sequences.append(("aug_31", iaa.Sequential([iaa.Add(-5, per_channel=True), iaa.Emboss(alpha=0.5)])))
        elif (aug == 31):
            sequences.append(("aug_32", iaa.Sequential([iaa.Add(-5, per_channel=True), iaa.Emboss(alpha=1.0)])))
        elif (aug == 32):
            sequences.append(("aug_33", iaa.Sequential([iaa.Add(-5, per_channel=True), iaa.Emboss(alpha=0.75)])))
        elif (aug == 33):
            sequences.append(("aug_34", iaa.Sequential([iaa.Add(0, per_channel=True), iaa.GaussianBlur(1.0)])))
        elif (aug == 34):
            sequences.append(("aug_35", iaa.Sequential([iaa.Add(0, per_channel=True), iaa.GaussianBlur(1.25)])))
        elif (aug == 35):
            sequences.append(("aug_36", iaa.Sequential([iaa.Add(0, per_channel=True), iaa.GaussianBlur(1.5)])))
        elif (aug == 36):
            sequences.append(("aug_37", iaa.Sequential([iaa.Add(5, per_channel=True), iaa.GaussianBlur(1.0)])))
        elif (aug == 37):
            sequences.append(("aug_38", iaa.Sequential([iaa.Add(5, per_channel=True), iaa.GaussianBlur(1.25)])))
        elif (aug == 38):
            sequences.append(("aug_39", iaa.Sequential([iaa.Add(5, per_channel=True), iaa.GaussianBlur(1.5)])))
        elif (aug == 39):
            sequences.append(("aug_40", iaa.Sequential([iaa.Add(-5, per_channel=True), iaa.GaussianBlur(1.0)])))
        elif (aug == 40):
            sequences.append(("aug_41", iaa.Sequential([iaa.Add(-5, per_channel=True), iaa.GaussianBlur(1.25)])))
        for prefix, seq in sequences:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            cv2.imwrite(img_path, seq.augment_image(img))
        return img_path
