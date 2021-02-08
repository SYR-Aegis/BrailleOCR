import cv2
import argparse
import numpy as np
import os
import glob
import sys

from PIL import ImageFont, ImageDraw, Image
from pathlib import Path

# a function to draw training images
def draw_image(textlist, TLGAN_save_path, CRNN_save_path):

    # make sure that the save path exists
    Path(TLGAN_save_path).mkdir(parents=True, exist_ok=True)
    Path(CRNN_save_path).mkdir(parents=True, exist_ok=True)

    full_img = np.zeros((0, 256, 3), dtype=np.uint8)
    bounding_boxes = []
    crnn_dataframe = {}

    for text, i in zip(textlist, range(len(textlist))):

        if len(text) <= 10:
            img = np.zeros((60, 256, 3), dtype=np.uint8)
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            draw.text((30, 20), text, font=font, fill=(b, g, r))

            #draw CRNN image
            cv2.imwrite(os.path.join(CRNN_save_path, str(len(os.listdir(CRNN_save_path))) + ".jpg"), np.array(img_pil))
            crnn_dataframe.update({str(len(os.listdir(CRNN_save_path))-1) + ".jpg": text})

            full_img = np.concatenate([full_img, img_pil], axis=0)

            xmin = 30
            xmax = xmin + 20*len(text)
            ymin = 20 + 60*i
            ymax = ymin+20

            bounding_boxes.append([xmin, xmax, ymin, ymax])
        else:
            full_img = np.concatenate([full_img, np.zeros((60, 256, 3))], axis=0)

    # make sure that the images are in the same size
    if len(textlist) < args.n_text:
        full_img = np.concatenate([full_img, np.zeros((60*(args.n_text-len(textlist)), 256, 3),)], axis=0)

    # check that images are 256X256
    if full_img.shape[0] < 256:
        full_img = np.concatenate([full_img, np.zeros(((256-full_img.shape[0]), 256, 3))], axis=0)

    cv2.imwrite(os.path.join(TLGAN_save_path, str(len(os.listdir(TLGAN_save_path))) + ".jpg"), full_img)

    return {str(len(os.listdir(TLGAN_save_path))-1) + ".jpg": bounding_boxes}, crnn_dataframe


def writeCSV(TLGAN_dataframe, CRNN_dataframe, TLGAN_csv_filename="TLGAN.csv", CRNN_csv_filename="CRNN.csv"):

    with open(TLGAN_csv_filename, "w") as f:
        cnt = 0
        for filename in TLGAN_dataframe.keys():
            print(f"writing CSV ... {cnt}/{len(TLGAN_dataframe.keys())} done")
            f.write(filename)
            for bb in TLGAN_dataframe[filename]:
                for axis in bb:
                    f.write(","+str(axis))
            f.write("\n")
            cnt += 1

    with open(CRNN_csv_filename, 'w', encoding="utf-8") as f:
        cnt = 0
        for filename in CRNN_dataframe.keys():
            print(f"writing CSV ... {cnt}/{len(CRNN_dataframe.keys())} done")
            f.write(filename)
            for label in CRNN_dataframe[filename]:
                f.write(","+str(label))
            f.write("\n")
            cnt += 1


if __name__ == '__main__':
    # new_crawling_for_datasets
    # sample_text
    parser = argparse.ArgumentParser(description="Convert text to image file")
    parser.add_argument("--text_file_path", type=str, default="./")
    parser.add_argument("--text_file_name", type=str, default="new_crawling_for_datasets.txt")
    parser.add_argument("--TLGAN_save_path", type=str, default="./images/TLGAN")
    parser.add_argument("--CRNN_save_path", type=str, default="./images/CRNN")
    parser.add_argument("--n_text", type=int, default=4)
    parser.add_argument("--simple", type=bool, default=True)
    parser.add_argument("--n_simple", type=int, default=50)

    args = parser.parse_args()
    n_text = args.n_text

    tlgan_csv = {}
    crnn_csv = {}
    is_simple = args.simple
    n_simple = args.n_simple

    if os.path.exists(args.TLGAN_save_path):
        for file in glob.glob(args.TLGAN_save_path+"/*"):
            os.remove(file)

    if os.path.exists(args.CRNN_save_path):
        for file in glob.glob(args.CRNN_save_path + "/*"):
            os.remove(file)

    if is_simple:

        with open(os.path.join(args.text_file_path, args.text_file_name), 'r', encoding="utf-8") as textFile:
            lines = textFile.read().split("\n")[:n_simple]

            b, g, r = 255, 255, 255
            fontpath = "fonts/H2GTRM.TTF"
            font = ImageFont.truetype(fontpath, 20)

            if len(lines) <= n_text:
                tlgan, crnn = draw_image(lines, args.TLGAN_save_path, args.CRNN_save_path)
                tlgan_csv.update(tlgan)
                crnn_csv.update(crnn)

            else:
                for i in range(0, len(lines), n_text):
                    print(f"writing images ... {i}/{len(lines)} done")

                    if i + n_text >= len(lines):
                        tlgan, crnn = draw_image(lines[i:], args.TLGAN_save_path, args.CRNN_save_path)
                        tlgan_csv.update(tlgan)
                        crnn_csv.update(crnn)
                    else:
                        tlgan, crnn = draw_image(lines[i:i + n_text], args.TLGAN_save_path, args.CRNN_save_path)
                        tlgan_csv.update(tlgan)
                        crnn_csv.update(crnn)

        writeCSV(tlgan_csv, crnn_csv)

    else:
        with open(os.path.join(args.text_file_path, args.text_file_name), 'r', encoding="utf-8") as textFile:
            lines = textFile.read().split("\n")

            b, g, r = 255, 255, 255
            fontpath = "fonts/H2GTRM.TTF"
            font = ImageFont.truetype(fontpath, 20)

            if len(lines) <= n_text:
                tlgan, crnn = draw_image(lines, args.TLGAN_save_path, args.CRNN_save_path)
                tlgan_csv.update(tlgan)
                crnn_csv.update(crnn)

            else:
                for i in range(0, len(lines), n_text):
                    print(f"writing images ... {i}/{len(lines)} done")

                    if i+n_text >= len(lines):
                        tlgan, crnn = draw_image(lines[i:], args.TLGAN_save_path, args.CRNN_save_path)
                        tlgan_csv.update(tlgan)
                        crnn_csv.update(crnn)

                    else:
                        tlgan, crnn = draw_image(lines[i:i+n_text], args.TLGAN_save_path, args.CRNN_save_path)
                        tlgan_csv.update(tlgan)
                        crnn_csv.update(crnn)
        writeCSV(tlgan_csv, crnn_csv)
