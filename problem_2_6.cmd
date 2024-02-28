:: This CMD script runs the seam carving for content-aware image resizing system
:: The goal is to form some perceptually pleasing outputs where rezising preserves content
:: as well as examples of images that lead to bad outcomes. 

::=====================
::Programming Problem 6
::=====================
@echo off

TITLE Seam Carving: Content-Aware Image Resizing System

ECHO Starting script... 

::-----> Bad Outcome Expected <----- 

ECHO Processing image 1 out of 5

python SeamCarvingReduceWidth.py -in input_img_rayo -out outputReduceWidthRayo -pix 150

python SeamCarvingReduceHeight.py -in input_img_rayo -out outputReduceHeightRayo -pix 150

::=====================

ECHO Processing image 2 out of 5

python SeamCarvingReduceWidth.py -in input_img_smiley -out outputReduceWidthSmiley -pix 200

python SeamCarvingReduceHeight.py -in input_img_smiley -out outputReduceHeightSimiley -pix 200

::=====================

ECHO Processing image 3 out of 5

python SeamCarvingReduceWidth.py -in input_img_giza -out outputReduceWidthGiza -pix 400

python SeamCarvingReduceHeight.py -in input_img_giza -out outputReduceHeightGiza -pix 400

::=====================

::-----> Good Outcome Expected <----- 

ECHO Processing image 4 out of 5

python SeamCarvingReduceWidth.py -in input_img_caracas -out outputReduceWidthCaracas -pix 250

python SeamCarvingReduceHeight.py -in input_img_caracas -out outputReduceHeightCaracas -pix 250

::=====================

ECHO Processing image 5 out of 5

python SeamCarvingReduceWidth.py -in input_img_totoro -out outputReduceWidthTotoro -pix 250

python SeamCarvingReduceHeight.py -in input_img_totoro -out outputReduceHeightTotoro -pix 250

::=====================