import cv2
import numpy as np
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

from PIL import Image

# method :: 'pyplot', 'cv', 'pil'

def DrawBBoxes(img, bboxes, class_mapping, method='pyplot'):
    if method == 'cv':
        return DrawBBoxesUsingCV(img, bboxes, class_mapping)
    elif method == 'pil':
        return DrawBBoxesUsingPIL(img, bboxes, class_mapping)

def DrawBBoxesUsingCV(img, bboxes, class_mapping):
    class_to_color = {v: np.random.randint(0, 255, 3) for v in class_mapping}

    for bbox in bboxes:
        class_name = bbox['class']
        cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']),
                     (int(class_to_color[class_name][0]),
                      int(class_to_color[class_name][1]),
                      int(class_to_color[class_name][2])), 2)
        (retval,baseLine) = cv2.getTextSize(class_name,cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        textOrg = (bbox['x1'], bbox['y1'] - 0)

        # cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
        cv2.rectangle(img,
                      (textOrg[0], textOrg[1]+baseLine),
                      (textOrg[0]+retval[0], textOrg[1]-retval[1]),
                      (int(class_to_color[class_name][0]),
                      int(class_to_color[class_name][1]),
                      int(class_to_color[class_name][2])), -1)
        cv2.putText(img, class_name, textOrg, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return img


STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

def DrawBBoxesUsingPIL(img, bboxes, class_mapping):
    box_to_color_map = {v: STANDARD_COLORS[class_mapping[v] % len(STANDARD_COLORS)] for v in class_mapping}

    draw = ImageDraw.Draw(img)

    for bbox in bboxes:
        color = box_to_color_map[bbox['class']]
        (left, right, top, bottom) = (bbox['x1'], bbox['x2'], bbox['y1'], bbox['y2'])
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=2, fill=color)

        try:
            font = ImageFont.truetype('arial.ttf', 24)
        except IOError:
            font = ImageFont.load_default()

        display_str_list = [bbox['class']]
        display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
        # Each display_str has a top and bottom margin of 0.05x.
        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)
        if top > total_display_str_height:
            text_bottom = top
        else:
            text_bottom = bottom + total_display_str_height

        for display_str in display_str_list[::-1]:
            text_width, text_height = font.getsize(display_str)
            margin = np.ceil(0.05 * text_height)
            draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                            (left + text_width, text_bottom)], fill=color)
            draw.text((left + margin, text_bottom - text_height - margin),
                      display_str,
                      fill='black',
                      font=font)

            text_bottom -= text_height - 2 * margin

    return img

