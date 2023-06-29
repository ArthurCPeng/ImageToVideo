# Code to convert a video into a worldbox map-styled video. 
# Adapted from ImageToMap++ code by Author: Destring and Upgrade Author: IgnizHerz / Igniz

import math
import zlib
import base64
import os
from PIL import Image
import numpy as np
import cv2

TILES = {
    63: "mountains",
    62: "mountains",
    61: "mountains",
    60: "mountains",
    59: "mountains",
    58: "mountains",
    57: "mountains",
    56: "mountains",
    55: "mountains",
    54: "mountains",
    53: "mountains",
    52: "mountains",
    51: "mountains",
    50: "mountains",
    49: "mountains",
    48: "mountains",
    47: "mountains",
    46: "mountains",
    45: "mountains",
    44: "mountains",
    43: "mountains",
    42: "mountains",
    41: "mountains",
    40: "mountains",
    39: "mountains",
    38: "soil_high:wasteland_high",
    37: "soil_low:wasteland_low",
    36: "soil_high:swamp_high",
    35: "soil_low:swamp_low",
    34: "soil_high:jungle_high",
    33: "soil_low:jungle_low",
    32: "soil_high:infernal_high",
    31: "soil_low:infernal_low",
    30: "soil_high:mushroom_high",
    29: "soil_low:mushroom_low",
    28: "soil_high:enchanted_high",
    27: "soil_low:enchanted_low",
    26: "soil_high:savanna_high",
    25: "soil_low:savanna_low",
    24: "soil_high:corrupted_high",
    23: "soil_low:corrupted_low",
    22: "soil_high:field",
    21: "soil_low:road",
    20: "lava1",
    19: "soil_low:fireworks",
    18: "lava2",
    17: "soil_high:tnt", 
    16: "lava3",
    15: "close_ocean",
    14: "deep_ocean",
    13: "soil_high:grass_high",
    12: "soil_high",
    11: "soil_high:snow_high",
    10: "hills",
    9: "hills:snow_hills",
    8: "mountains",
    7: "mountains:snow_block",
    6: "sand",
    5: "sand:snow_sand",
    4: "shallow_waters",
    3: "shallow_waters:ice",
    2: "soil_low",
    1: "soil_low:snow_low",
    0: "soil_low:grass_low"
}

TILE_COLORS = {
    63: "414545",
    62: "414545",
    61: "414545",
    60: "414545",
    59: "414545",
    58: "414545",
    57: "414545",
    56: "414545",
    55: "414545",
    54: "414545",
    53: "414545",
    52: "414545",
    51: "414545",
    50: "414545",
    49: "414545",
    48: "414545",
    47: "414545",
    46: "414545",
    45: "414545",
    44: "414545",
    43: "414545",
    42: "414545",
    41: "414545",
    40: "414545",
    39: "414545",
    38: "6C7759",
    37: "849371",
    36: "453E34",
    35: "4D483E",
    34: "1F7020",
    33: "46A052",
    32: "68372D",
    31: "9C3626",
    30: "556338",
    29: "677642",
    28: "76B153",
    27: "8CDC6A",
    26: "CF931B",
    25: "F0B121",
    24: "533F51",
    23: "6F556C",
    22: "A8663A",
    21: "C1997C",
    20: "FF6700",
    19: "B43DCC",
    18: "FFAC00",
    17: "A30000", 
    16: "FFDE00",
    15: "4084E2",
    14: "3370CC",
    13: "5F833C",
    12: "B66F3A",
    11: "B4CFE5", #permafrost_high
    10: "5B5E5C",
    9: "E2EDEC",
    8: "414545",
    7: "FCFDFD",
    6: "F7E898",
    5: "AFF5F1",
    4: "55AEF0",
    3: "A7D6F4",
    2: "E2934B",
    1: "99BCDB",
    0: "99BCDB" 


}
COLORS = [126, 175, 70, 186, 213, 211, 213, 142, 18, 167, 214, 244, 85, 174, 240, 
          175, 245, 241, 247, 232, 152, 252, 253, 253, 69, 69, 69, 226, 237, 236,
          82, 82, 82, 211, 228, 227, 182, 111, 58, 84, 114, 45, 51, 112, 204,
          64, 132, 226, 255, 222, 0, 163, 0, 0, 255, 172, 0, 180, 61, 204,
	  255, 103, 0, 193, 153, 124, 168, 102, 58, 111, 85, 108, 83, 63, 81, 
          240, 177, 33, 207, 147, 27, 140, 220, 106, 118, 177, 83, 103, 118, 66,
          85, 99, 56, 156, 54, 38, 104, 55, 45, 70, 160, 82, 31, 112, 32,
          80, 129, 108, 106, 166, 139, 132, 147, 113, 108, 119, 89, 252, 253, 253,
          252, 253, 253, 252, 253, 253, 252, 253, 253, 252, 253, 253, 252, 253, 253, 
          252, 253, 253, 252, 253, 253, 252, 253, 253, 252, 253, 253, 252, 253, 253,
          252, 253, 253, 252, 253, 253, 252, 253, 253, 252, 253, 253, 252, 253, 253, 252, 253, 253, 252, 253, 253, 252, 253, 253, 252, 253, 253, 252, 253, 253, 252, 253, 253, 252, 253, 253, 252, 253, 253, 252, 253, 253]
CHUNK_SIZE = 64
COLOR_PALETTE = Image.new('P', (1, 1))
COLOR_PALETTE.putpalette(COLORS * 4)


def hex2rgb(hex_color):
    #return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

hex2rgb_converter = lambda x: hex2rgb(TILE_COLORS[x])
hex2rgb_vfunc = np.vectorize(hex2rgb_converter)
    
def quantize_with_palette(image, palette, dither):
    size = image.size
    chunks_width = size[0] // CHUNK_SIZE
    chunks_height = size[1] // CHUNK_SIZE
    chunks_ratio = chunks_height / chunks_width
    chunks_ratio_rev = chunks_width / chunks_height

    if add_bg:
        chunks_height = 8

    #size = (chunks_width * CHUNK_SIZE, math.ceil(chunks_ratio * chunks_width * CHUNK_SIZE))
    size = (math.ceil(chunks_height * CHUNK_SIZE * chunks_ratio_rev), chunks_height * CHUNK_SIZE)

    image = image.resize(size, resample=Image.Resampling.NEAREST)
    converted_image = image.im.convert("P", dither, palette.im)
    converted_image = image._new(converted_image)

    return converted_image


def image_to_map(image):

    px = 2
    
    if add_bg:
        px = 2
        
    w = image.size[0]
    h = image.size[1]
    data = image.getdata()

    data = list(data)
    map_img = np.array(data)
    map_img = map_img.reshape((h,w))

    map_img_final = np.zeros((h*px, w*px, 3))

    for i in range(h):
        for j in range(w):
            tile = map_img[i,j]
            color = hex2rgb(TILE_COLORS[tile])

            map_img_final[i*px:(i+1)*px,j*px:(j+1)*px,:] = color

    if add_bg:
        #bg = cv2.imread('bg.png')
        bg_h = bg.shape[0]
        bg_w = bg.shape[1]

        bg[610-round(h*px/2):610+round(h*px/2), bg_w//2-round(w*px/2):bg_w//2+round(w*px/2),:] = map_img_final

        if video_mode:
            return bg
        else:
            cv2.imwrite('test_bg.png', bg)
            return

    
    cv2.imwrite('test.png', map_img_final)
            
            
    
def read_map_array(image_array):
    image_file = Image.fromarray(image_array)
    #dithering = 1 if messagebox.askquestion("Dithering", f"Use dithering?\nMap size: {image_file.size[0] // 64 if image_file.size[0] // 64 <= 20 else 20}x{image_file.size[1] // 64}") == 'yes' else 0
    dithering=0
    converted_image = quantize_with_palette(image_file, COLOR_PALETTE, dithering)
    return image_to_map(converted_image)



def read_map_file(file_path):
    with Image.open(file_path) as image_file:
        #dithering = 1 if messagebox.askquestion("Dithering", f"Use dithering?\nMap size: {image_file.size[0] // 64 if image_file.size[0] // 64 <= 20 else 20}x{image_file.size[1] // 64}") == 'yes' else 0
        dithering=0
        converted_image = quantize_with_palette(image_file, COLOR_PALETTE, dithering)
        image_to_map(converted_image)


#Convert image
'''
add_bg = True
video_mode = False

path = 'avatar.png'
read_map_file(path)
'''

#Convert video
add_bg = True
video_mode = True
alert_freq = 20

video_path = 'testing.mov'
output_path = 'output.mp4'
background_path = 'bg.png'

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

print("FPS: {}".format(fps))
print("Total Frames: {}".format(total_frames))

bg = cv2.imread(background_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter(output_path,
                         fourcc,
                         fps, (bg.shape[1], bg.shape[0]))

no_frames = 0    
while cap.isOpened():
    ret, frame = cap.read()
    
    if no_frames % alert_freq == alert_freq-1:
        print("--> {}/{} frames completed".format(no_frames+1, total_frames))
        
    if not ret:
        break
    '''
    if no_frames % freq != 0:
        no_frames += 1
        continue
    '''

    #cv2.imwrite('frames_rickroll/{}.jpg'.format(pad(no_frames)), frame)
    #frame_small = cv2.resize(frame, (960, 540))
    #frame_small = frame_small[:,135:825,:]
    #cv2.imwrite('frames_rickroll_small/{}.jpg'.format(pad(no_frames)), frame_small)

    frame = frame[:,:,::-1]

    frame_map = read_map_array(frame)

    output.write(frame_map)

    no_frames += 1

    #if no_frames > 40:
        #break

output.release()
