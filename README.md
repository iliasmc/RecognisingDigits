<h1>Basic setup</h1>

1. Set **directoryLocation** *(line 9)* to the directory of the image you want the CNN to predict.

2. Set the boolean **show_testing** *(line 12)* to *False* if you want to see the prediction of only your own image.<br>
   Set the boolean **show_testing** *(line 12)* to *True* if you want to see predictions of random MNIST images.

3. Set the boolean **display_only_image** *(line 15)* to *True* if you only want to see your image and the CNN's prediction of it.<br>
   Set the boolean **display_only_image** *(line 15)* to *False* if you want to also see 25 random MNIST images and their predictions.

<h1>Changing your image in real time</h1>

In order to change your image and see its new prediction:<br>

>1. Run the program.
>2. Open an image editor (ideally Paint 3D).
>3. Use a brush of 3px and draw your digit.
>4. Save the image you've drawn.
>5. Close the current window of your running program to refresh the image and predict your new image.
>6. See the prediction of the image you've just drawn!
>7. Repeat steps 3, 4, 5 to change see the CNN's prediction of your newly edited image!

To terminate the program either use Task Manager or the IDE you're using the run the program.
