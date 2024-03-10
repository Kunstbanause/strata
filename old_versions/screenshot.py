import mss.tools

def Screenshot():
    with mss.mss() as sct:
        # The screen part to capture
        monitor = {"top": 40, "left": 20, "width": 350, "height": 550}
        #output = "screen.png".format(**monitor)

        # Grab the data
        sct_img = sct.grab(monitor)

        # Save to the picture file
        #mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
        return sct_img