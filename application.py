import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import base64

# utils
from utils import image_read_and_resize

# model
from model import generate_neural_style_transfer

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

TARGET_INPUT_IMAGE_SIZE = (300, 240)
obama_image_path = "sample_data\content_pictures\content1.jpg"
dog_image_path = "sample_data\content_pictures\content2.jpg"
bear_image_path = "sample_data\content_pictures\content3.jpg"

style1_image_path = "sample_data\style_pictures\style1.jpg"
style2_image_path = "sample_data\style_pictures\style2.jpg"
style3_image_path = "sample_data\style_pictures\style3.jpg"

sample_output_image = "generated_images\content_0_ style_0\epoch_1000.jpg"

obama_image_encoded = image_read_and_resize(
    obama_image_path, target_size=TARGET_INPUT_IMAGE_SIZE, input_type='filepath')
dog_image_encoded = image_read_and_resize(
    dog_image_path, target_size=TARGET_INPUT_IMAGE_SIZE, input_type='filepath')
bear_image_encoded = image_read_and_resize(
    bear_image_path, target_size=TARGET_INPUT_IMAGE_SIZE, input_type='filepath')

style1_image_encoded = image_read_and_resize(
    style1_image_path, target_size=TARGET_INPUT_IMAGE_SIZE, input_type='filepath')
style2_image_encoded = image_read_and_resize(
    style2_image_path, target_size=TARGET_INPUT_IMAGE_SIZE, input_type='filepath')
style3_image_encoded = image_read_and_resize(
    style3_image_path, target_size=TARGET_INPUT_IMAGE_SIZE, input_type='filepath')

sample_output_image_encoded = image_read_and_resize(
    sample_output_image, target_size=TARGET_INPUT_IMAGE_SIZE, input_type='filepath'
)

upload_image_style = {
    'height': "40px",
    'lineHeight': "40px",
    'borderWidth': '1px',
    'borderStyle': 'dashed',
    'borderRadius': '5px',
    'textAlign': 'center',
    'margin': '10px'
}


app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("NEURAL STYLE TRANSFER"),
            className='text-align-center text-primary mb-2', width={"size": 6, "offset": 1}), justify='center'),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H4("Content Image", className="text-center")),
            dbc.CardBody([
                dcc.Upload(children=["Drag and drop or ",
                                     html.A("Select Files", className='font-weight-bold text-primary')],
                           id="content_img_upload", style=upload_image_style, multiple=False),
                html.Img(id='content_image',
                         src="data:image/png;base64,{}".format(
                             obama_image_encoded.decode()),
                         height=300, width=240, style={"display": "block", "margin-left": "auto", "margin-right": "auto"}),
                dbc.ButtonGroup([
                    dbc.Button("Obama", id="obama_btn", color='primary'),
                    dbc.Button("Dog", id="dog_btn", color='primary'),
                    dbc.Button("Bear", id="bear_btn", color='primary')
                ], style={"margin-top": "5px"})
            ], className='text-center')
        ]), xl=6, lg=6, md=12, sm=12, xs=12),
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H4("Style Image", className="text-center")),
            dbc.CardBody([
                dcc.Upload(children=["Drag and drop or ",
                                     html.A("Select Files", className='font-weight-bold text-primary')],
                           id="style_img_upload", style=upload_image_style, multiple=False),
                html.Img(id='style_image',
                         src="data:image/png;base64,{}".format(
                             style1_image_encoded.decode()),
                         height=300, width=240, style={"display": "block", "margin-left": "auto", "margin-right": "auto"}),
                dbc.ButtonGroup([
                    dbc.Button("style_1", id="style_1_btn", color='primary'),
                    dbc.Button("style_2", id="style_2_btn", color='primary'),
                    dbc.Button("style_3", id="style_3_btn", color='primary')
                ], style={"margin-top": "5px"})
            ], className='text-center')
        ]), xl=6, lg=6, md=12, sm=12, xs=12)]),
    dbc.Row(dbc.Col([

        dbc.Button(["Generate"], id="generate_btn", className='me-1 mt-4 mb-4',
                   style={'width': "30%", "height": "50px"}),
        dbc.Spinner(html.Div(id="generating-image")),
        dbc.Card([
            dbc.CardHeader(html.H4("Generated image")),
            dbc.CardBody(
                html.Img(id='generated_image',
                         src="data:image/png;base64,{}".format(
                             sample_output_image_encoded.decode()
                         ), height=300, width=240, style={'display': 'block', "margin-left": "auto", "margin-right": "auto"}),
            )
        ], className="text-center mt-3")
    ], width={'size': 6, 'offset': 3})),
    dcc.Store(id='content_store', data={
              'content_image': obama_image_encoded.decode()}),
    dcc.Store(id='style_store', data={
              'style_image':  style1_image_encoded.decode()
              })
])


@app.callback([Output("content_image", "src"), Output("content_store", "data")],
              [Input("obama_btn", "n_clicks"),
               Input("dog_btn", "n_clicks"),
               Input("bear_btn", "n_clicks"),
               Input("content_img_upload", "contents"),
               ],
              )
def show_content_img(obama_btn, dog_btn, bear_btn, contents):
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = "No clicks yet"
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    img = obama_image_encoded.decode()
    src = "data:image/png;base64,{}".format(img)

    if button_id == 'obama_btn':
        img = obama_image_encoded.decode()
        src = "data:image/png;base64,{}".format(img)
    elif button_id == 'dog_btn':
        img = dog_image_encoded.decode()
        src = "data:image/png;base64,{}".format(img)
    elif button_id == 'bear_btn':
        img = bear_image_encoded.decode()
        src = "data:image/png;base64,{}".format(img)

    elif button_id == 'content_img_upload':
        content_type, content_string = contents.split(',')
        content_img = image_read_and_resize(
            content_string, input_type='base64')
        img = content_img.decode()
        src = "data:image/png;base64,{}".format(img)

    return src, {"content_image": img}


@app.callback([Output("style_image", "src"), Output("style_store", "data")],
              [Input("style_1_btn", "n_clicks"),
               Input("style_2_btn", "n_clicks"),
               Input("style_3_btn", "n_clicks"),
               Input("style_img_upload", "contents"), ]
              )
def show_style_img(btn1, btn_2, btn_3, contents):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = "No clicks yet"
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    img = style1_image_encoded.decode()
    src = "data:image/png;base64,{}".format(img)

    if button_id == 'style_1_btn':
        img = style1_image_encoded.decode()
        src = "data:image/png;base64,{}".format(img)
    elif button_id == 'style_2_btn':
        img = style2_image_encoded.decode()
        src = "data:image/png;base64,{}".format(img)
    elif button_id == 'style_3_btn':
        img = style3_image_encoded.decode()
        src = "data:image/png;base64,{}".format(img)
    elif button_id == "style_img_upload":
        content_type, content_string = contents.split(',')
        content_img = image_read_and_resize(
            content_string, input_type='base64')
        img = content_img.decode()
        src = "data:image/png;base64,{}".format(img)

    return src, {"style_image": img}


@app.callback([Output("generated_image", 'src'), Output("generating-image", "children")],
              [Input("generate_btn", 'n_clicks'),
               State("content_store", "data"),
               State("style_store", "data")], prevent_initial_call=True)
def generate_style_transfer(btn_1, content_img, style_img):
    # content_data / style_data byte_string
    config = {
        "total_variation_weight": 30,
        "style_weight": 1e-2,
        "content_weight": 1e4,
        "image_size": (128, 128),
        "epochs": 20
    }

    base64_image = generate_neural_style_transfer(
        content_img['content_image'], style_img['style_image'], config)

    return "data:image/png;base64,{}".format(base64_image.decode()), html.H5("Image Generated")


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=False)
