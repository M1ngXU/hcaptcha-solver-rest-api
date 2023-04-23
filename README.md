# HCaptcha Solver Rest API
github: <https://github.com/M1ngXU/hcaptcha-solver-rest-api>
<br>
docs: <https://hcaptcha-solver.mintlify.app/>

This rest API solves hcaptcha challenges by using AI image-recognition models from [this](https://github.com/QIN2DIM/hcaptcha-challenger) repository. It doesn't collect any data nor requires some kind of API tokens.

## Endpoints
Breaking changes will be on a new endpoint while the old endpoint will still be available.

### `/v0`

Submit a POST request to this URI to use the API; following JSON-payload is required:
```json
{
    "prompt": "The prompt of your challenge",
    "images": ["123/abc", "..."]
}
```
- `prompt`: The prompt of your challenge.
- `images`: The id of the images of your challenge. If your image has an url of `https://imgs.hcaptcha.com/123/abc`, then your id is `123/abc`.

#### Output
- `200`:
```json
{
    "trues": ["123/abc", "..."], // the id of all "true" images
    "errors": {"256/abc": "error"} // Errors for specific images, a map from id to error.
}
```
- `400`: Something is wrong with your request. The error message is returned as the response body.
- `424`: Some problem with ort, the onnx runtime used. The error message is returned as the response body.
- `500`: An internal server error happened. The error message is returned as the response body.
- `501`: The prompt requires a model that hasn't been created yet. HCaptcha creates new challenges all the time (adding new objects) and models for these objects need to be created. Check out the status of some [issues](https://github.com/QIN2DIM/hcaptcha-challenger/labels/%F0%9F%94%A5%20challenge) in the repository containing the models.