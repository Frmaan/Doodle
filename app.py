import os
import gradio as gr
import numpy as np
from PIL import Image
from sketch_recognition.predict import predict_sketch

def preprocess_drawing(drawing):

    if drawing is None or drawing["composite"] is None:
        return None

    #get grayscale image
    image = drawing["composite"]

    #invert
    image = 255 - image

    #find pixels of the drawing
    coords = np.argwhere(image > 0)

    if len(coords) == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = image[y_min:y_max + 1, x_min:x_max + 1]

    height, width = cropped.shape

    margin = max(1, int(max(height,width) * 0.1))

    side = max(height, width) + (2 * margin)

    top = (side - height) // 2
    bottom = side - height - top

    left = (side - width) // 2
    right = side - width - left

    canvas = np.pad(
        cropped,
        ((top, bottom), (left, right)),
        mode="constant",
        constant_values=0
    )


    resized = Image.fromarray(canvas).resize(
        (28,28),
        Image.Resampling.LANCZOS
    )

    processed = np.asarray(resized, dtype=np.float32) / 255.0

    return processed

def inspect_drawing(drawing):

    processed = preprocess_drawing(drawing)

    if processed is None:
        return "Please draw something first."

    return (
        f"Preprocessing successfull!\n"
        f"Shape: {processed.shape}\n"
        f"Data Type: {processed.dtype}\n"
        f"Minumum pixel value: {processed.min()}\n"
        f"Maximum pixel value: {processed.max()}\n"
        f"Non-zero pixels: {np.count_nonzero(processed)}"
    )

def recognize_drawing(drawing):
    processed_image = preprocess_drawing(drawing)

    if processed_image is None:
        return None

    results = predict_sketch(processed_image)

    return results

with gr.Blocks(title="Sketch Recognition",
               css="""
                .sketchpad-container{
                    width: 100% !important;
                    max-width: 600px;
                    margin: auto;
                }
                """
               ) as app:

    gr.Markdown("# Sketch Recognition")
    gr.Markdown("Draw something and let our CNN identify it!")

    with gr.Column():
        sketchpad = gr.Sketchpad(
            label="Your drawing",
            height=600,
            width="100%",
            canvas_size=(800,800),
            fixed_canvas=True,
            container=False,
            image_mode="L",
            type="numpy"
        )

    guess_button = gr.Button("Guess", variant="primary",size="lg")

    result = gr.Label(label="Prediction", num_top_classes=5)


    guess_button.click(
        fn=recognize_drawing,
        inputs=sketchpad,
        outputs=result
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    app.launch(
        server_name="0.0.0.0",
        server_port= port
    )
