from fastai.vision.all import *
from fastai.vision.widgets import *
from fastdownload import download_url
from ddgs import DDGS

### Example How to use Duck gogo to search images
# ims = DDGS().images('grizzly bear')
# len(ims)
# ims = ['http://3.bp.blogspot.com/-S1scRCkI3vY/UHzV2kucsPI/AAAAAAAAA-k/YQ5UzHEm9Ss/s1600/Grizzly%2BBear%2BWildlife.jpg']
# dest = 'images/grizzly.jpg'
# download_url(ims[0], dest)

### Example show image thumb in notebook
# im = PILImage.create(dest)
# im.to_thumb(128,128)

# Define types over iterate 
bear_types = 'grizzly','black','teddy'

# Define data folder
path = Path('bears')

# Download images
if not path.exists():
    path.mkdir()
    for o in bear_types:
        dest = (path/o)
        dest.mkdir(exist_ok=True)

        queries = [f'{o} bear', f'{o} bear wild', f'{o} bear photo']
        if o == 'teddy':
            queries = [f'{o} bear', f'{o} bear peluche',]
        urls = []
        for q in queries:
            results = DDGS().images(q, max_results=150, safesearch='off')
            urls += [item["image"] for item in results]
            
        # deduplicate
        urls = list(set(urls))
        download_images(dest, urls=urls)

# Get Images Files
fns = get_image_files(path)
len(fns)

# Verify broken images
failed = verify_images(fns)
failed

# Remove broken images
failed.map(Path.unlink);

# Create Data Loaders
# Indipendent Var -> Image | Depentend Var -> Category (Our Prediction)
bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))

# Removed wrap to prevent it use gpu (my current pc can't use gpu)
bears = bears.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms(mult=2, max_warp=0.))
dls = bears.dataloaders(path)

# Show batch example
dls.valid.show_batch(max_n=4, nrows=2)    

# Fine tune resnet18
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)

# Test the model
categories = learn.dls.vocab
pred, idx, probs = learn.predict('grizzly.jpeg')
res = dict(zip(categories, map(float,probs)))
print(res)

### Plot confusion matrix graphic
# interp = ClassificationInterpretation.from_learner(learn)
# interp.plot_confusion_matrix()

### Plot the top 5 losses in the model
# interp.plot_top_losses(5, nrows=1)

### Fast AI Image cleaner to remove or change category of prediction in validation set
# cleaner = ImageClassifierCleaner(learn)
# cleaner

### First line delete image signed to deleted in previous tool
### Second lane change the category if it was asked in the previous step
# for idx in cleaner.delete(): cleaner.fns[idx].unlink()
# for idx,cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/cat)

### Export model to a pkl file named export
learn.export()