import cv2
import argparse
import numpy as np
import os

from PIL import ImageFont, ImageDraw, Image


# a function to draw training images
def draw_image(textlist, save_path):

    full_img = np.zeros((0, 256, 3), dtype=np.uint8)
    bounding_boxes = []

    for text, i in zip(textlist, range(len(textlist))):

        img = np.zeros((60, 256, 3), dtype=np.uint8)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((30, 20), text, font=font, fill=(b, g, r))


        full_img = np.concatenate([full_img, img_pil], axis=0)

        xmin = 30
        xmax = xmin + 20*len(text)
        ymin = 20 + 60*i
        ymax = ymin+20

        bounding_boxes.append([xmin, xmax, ymin, ymax])


    # make sure that the images are in the same size
    if len(textlist) < args.n_text:
        full_img = np.concatenate([full_img, np.zeros((60*(args.n_text-len(textlist)), 256, 3),)], axis=0)

    # make sure that the save path exists
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    cv2.imwrite(os.path.join(save_path, str(len(os.listdir(save_path)))+".jpg"), full_img)

    return {str(len(os.listdir(save_path)))+".jpg": bounding_boxes}


def writeCSV(dataframe, csv_filename="TLGAN.csv"):

    with open(csv_filename, "w") as f:
        for filename in dataframe.keys():
            f.write(filename)
            for bb in dataframe[filename]:
                f.write(","+str(bb))
            f.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert text to image file")
    parser.add_argument("--text_file_path", type=str, default="./")
    parser.add_argument("--text_file_name", type=str, default="sample_text.txt")
    parser.add_argument("--save_path", type=str, default="./images/")
    parser.add_argument("--n_text", type=int, default=4)

    args = parser.parse_args()
    n_text = args.n_text

    csv = {}

    with open(os.path.join(args.text_file_path, args.text_file_name), 'r', encoding="utf-8") as textFile:
        lines = textFile.read().split("\n")

        b, g, r = 255, 255, 255
        fontpath = "fonts/H2GTRM.TTF"
        font = ImageFont.truetype(fontpath, 20)

        if len(lines) <= n_text:
            csv.update(draw_image(lines, args.save_path))

        else:
            for i in range(0, len(lines), n_text):

                if i+n_text >= len(lines):
                    csv.update(draw_image(lines[i:], args.save_path))

                else:
                    csv.update(draw_image(lines[i:i+n_text], args.save_path))
    writeCSV(csv)
