import pandas as pd
from pathlib import Path
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def load_paths(root='ISIC2018'):

    image_root = root / Path(r'ISIC2018_Task1-2_Training_Input')
    task1_root = root / Path(r'ISIC2018_Task1_Training_GroundTruth')
    taks2_root = root / Path(r'ISIC2018_Task2_Training_GroundTruth_v3')

    df = pd.DataFrame(dict(image_path = list(image_root.rglob("*.jpg"))))
    df['image_id'] = df['image_path'].apply(lambda x: x.stem)
    df['image_filename'] = df['image_path'].apply(lambda x: x.name)

    temp_dict = {temp.stem: temp for temp in list(task1_root.rglob("*.png"))}
    df['task1_path'] = df['image_id'].apply(lambda x: temp_dict[x + '_segmentation'])
    df['task1_filename'] = df['task1_path'].apply(lambda x: x.name)

    df = df.sort_values(by=['image_id'])

    return df

def make_split(data_df, train_size=2080, valid_size=264):
    test_size = len(data_df)-valid_size-train_size
    assert test_size>0, "Oh no! The test_size is negative!"
    
    train_df, valid_df = train_test_split(data_df, train_size=train_size, random_state = 999)
    test_df, valid_df = train_test_split(valid_df, train_size=test_size, random_state = 999)
    train_df, valid_df, test_df = train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)

    return train_df, valid_df, test_df

def make_generators(data_gen_args, mask_gen_args, image_flow_from_dataframe_args, mask_flow_from_dataframe_args, train_df, valid_df, test_df):
    img_idg = ImageDataGenerator(**data_gen_args)
    mask_idg = ImageDataGenerator(**mask_gen_args)

    def make_generator(in_df, img_idg, mask_idg, image_args, mask_args, root='ISIC2018'):
        image_generator = img_idg.flow_from_dataframe(in_df, root / Path(r'ISIC2018_Task1-2_Training_Input'), **image_args)
        mask_generator = mask_idg.flow_from_dataframe(in_df, root / Path(r'ISIC2018_Task1_Training_GroundTruth'), **mask_args)
        return zip(image_generator, mask_generator)

    train_generator = make_generator(train_df, img_idg, mask_idg, image_flow_from_dataframe_args, mask_flow_from_dataframe_args)
    valid_generator = make_generator(valid_df, img_idg, mask_idg, image_flow_from_dataframe_args, mask_flow_from_dataframe_args)
    test_generator  = make_generator(test_df, img_idg, mask_idg, image_flow_from_dataframe_args, mask_flow_from_dataframe_args)

    return train_generator, valid_generator, test_generator