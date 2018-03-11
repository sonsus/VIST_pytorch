from PIL import Image as Img #load_img(), is_image_file

def load_img(filepath):
    img=Img.opne(filepath).convert("RGB")
    img=img.resize((224,224), Img.BICUBIC)
    return img

def is_imgfile(filepath):
    return any(filepath.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".gif"])