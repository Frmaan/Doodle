import os
import html
import gradio as gr
import numpy as np
from sketch_recognition.predict import predict_sketch
from sketch_recognition.preprocessing import preprocess_drawing

# DRAWING INSPECTION


def inspect_drawing(drawing):

    print("Input type:", type(drawing))

    if isinstance(drawing, dict):
        print("Dictionary keys:", drawing.keys())
    elif isinstance(drawing, np.ndarray):
        print("Array shape:", drawing.shape)
        print("Array dtype:", drawing.dtype)

    processed = preprocess_drawing(drawing)

    if processed is None:
        return "Please draw something first."

    return (
        f"Preprocessing successful!\n"
        f"Shape: {processed.shape}\n"
        f"Data Type: {processed.dtype}\n"
        f"Minimum pixel value: {processed.min()}\n"
        f"Maximum pixel value: {processed.max()}\n"
        f"Non-zero pixels: {np.count_nonzero(processed)}"
    )


# PREDICTION HTML


def create_prediction_html(results):

    if not results:
        return """
            <div class="prediction-heading">
                No prediction available.
            </div>
        """

    # Get top 3 predictions
    top_predictions = sorted(results.items(), key=lambda item: item[1], reverse=True)[
        :3
    ]

    prediction_html = """
        <div class="prediction-heading">
            Top predictions
        </div>
    """

    for name, score in top_predictions:

        name = html.escape(str(name))
        percentage = float(score) * 100
        bar_width = max(0, min(100, percentage))

        prediction_html += f"""
            <div class="prediction-row">

                <div class="prediction-info">
                    <span>{name}</span>
                    <span>{percentage:.1f}%</span>
                </div>

                <div class="prediction-track">
                    <div
                        class="prediction-bar"
                        style="width: {bar_width}%;">
                    </div>
                </div>

            </div>
        """

    return prediction_html


# SHARED PREDICTION LOGIC


def get_prediction(drawing):

    processed_image = preprocess_drawing(drawing)

    if processed_image is None:
        return """
            <div class="prediction-heading">
                Please draw something first!
            </div>
        """

    results = predict_sketch(processed_image)

    return create_prediction_html(results)


# DESKTOP PREDICTION HANDLER


def recognize_desktop(drawing):

    prediction_html = get_prediction(drawing)

    return (prediction_html, gr.update(visible=True), gr.update(visible=False))


# MOBILE PREDICTION HANDLER


def recognize_mobile(drawing):

    prediction_html = get_prediction(drawing)

    return (prediction_html, gr.update(visible=True), gr.update(visible=False))


# INTERFACE

with gr.Blocks(
    title="Doodle",
    css="""

    #title {
        text-align: center !important;
        margin-bottom: 20px !important;
    }


    #desktop-view {
        display: block;
        width: 100%;
        max-width: 1400px;
        margin: 0 auto;
    }

    #desktop-layout {
        align-items: stretch !important;
        gap: 16px !important;
    }

    #desktop-drawing {
        min-width: 0 !important;
    }

    #desktop-prediction-panel {
        min-width: 0 !important;
        background: var(--body-background-fill) !important;
        border: 1px solid var(--border-color-primary) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        justify-content: center !important;
        align-self: stretch !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12) !important;
    }

    #desktop-prediction-result {
        width: 100% !important;
        min-width: 0 !important;
    }

    #desktop-prediction-panel button {
        width: 100% !important;
        margin-top: 12px !important;
    }


    #mobile-view {
        display: none;
    }


    .prediction-heading {
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 14px;
    }

    .prediction-row {
        width: 100%;
        margin-bottom: 12px;
    }

    .prediction-info {
        display: flex;
        justify-content: space-between;
        width: 100%;
        font-size: 0.9rem;
        margin-bottom: 5px;
        gap: 8px;
    }

    .prediction-track {
        width: 100%;
        height: 6px;
        background: var(--border-color-primary);
        border-radius: 10px;
        overflow: hidden;
    }

    .prediction-bar {
        height: 100%;
        background: var(--button-primary-background-fill);
        border-radius: 10px;
    }


    #prediction-overlay {
        position: fixed !important;

        left: 50% !important;
        bottom: 12px !important;
        top: auto !important;

        transform: translateX(-50%) !important;

        width: calc(100% - 24px) !important;
        max-width: 560px !important;

        z-index: 10000 !important;

        background: var(--body-background-fill) !important;
        border: 1px solid var(--border-color-primary) !important;
        border-radius: 16px !important;

        padding: 12px !important;
        gap: 10px !important;

        box-shadow: 0 -4px 24px rgba(0, 0, 0, 0.2) !important;
    }

    #mobile-prediction-result {
        width: 100% !important;
        min-width: 0 !important;
    }

    #prediction-overlay button {
        width: auto !important;
        min-width: 105px !important;
        align-self: center !important;
    }


    @media (max-width: 640px) {

        #desktop-view {
            display: none !important;
        }

        #mobile-view {
            display: block !important;
            width: 100%;
        }

        #title {
            margin-bottom: 12px !important;
        }

        .prediction-info {
            font-size: 0.85rem;
        }

    }

    """,
) as app:

    # PROJECT TITLE

    gr.Markdown("# Doodle", elem_id="title")

    # DESKTOP LAYOUT

    with gr.Column(elem_id="desktop-view"):

        with gr.Row(elem_id="desktop-layout"):

            # Drawing area
            with gr.Column(scale=7, elem_id="desktop-drawing"):

                desktop_sketchpad = gr.Sketchpad(
                    label="Your drawing",
                    height="80vh",
                    width="100%",
                    canvas_size=(1600, 900),
                    fixed_canvas=True,
                    container=False,
                    image_mode="L",
                    type="numpy",
                )

                desktop_guess_button = gr.Button("Guess", variant="primary", size="lg")

            # Right-side prediction panel
            with gr.Column(
                scale=3, visible=False, elem_id="desktop-prediction-panel"
            ) as desktop_prediction_panel:

                desktop_result = gr.HTML(elem_id="desktop-prediction-result")

                desktop_close_button = gr.Button("Close", variant="primary", size="lg")

    # MOBILE LAYOUT

    with gr.Column(elem_id="mobile-view"):

        mobile_sketchpad = gr.Sketchpad(
            label="Your drawing",
            height="70vh",
            width="100%",
            canvas_size=(800, 1300),
            fixed_canvas=True,
            container=False,
            image_mode="L",
            type="numpy",
        )

        mobile_guess_button = gr.Button("Guess", variant="primary", size="lg")

    # MOBILE PREDICTION NOTIFICATION

    with gr.Column(
        visible=False, elem_id="prediction-overlay"
    ) as mobile_prediction_overlay:

        mobile_result = gr.HTML(elem_id="mobile-prediction-result")

        mobile_close_button = gr.Button("Close", variant="primary", size="lg")

    # desktop events

    desktop_guess_button.click(
        fn=recognize_desktop,
        inputs=desktop_sketchpad,
        outputs=[desktop_result, desktop_prediction_panel, mobile_prediction_overlay],
    )

    desktop_close_button.click(
        fn=lambda: gr.update(visible=False),
        inputs=None,
        outputs=desktop_prediction_panel,
    )

    # MOBILE EVENTS

    mobile_guess_button.click(
        fn=recognize_mobile,
        inputs=mobile_sketchpad,
        outputs=[mobile_result, mobile_prediction_overlay, desktop_prediction_panel],
    )

    mobile_close_button.click(
        fn=lambda: gr.update(visible=False),
        inputs=None,
        outputs=mobile_prediction_overlay,
    )

# launch

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    app.launch(
        server_name="0.0.0.0", server_port=port, footer_links=[], run_history=False
    )
