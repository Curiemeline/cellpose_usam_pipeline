import os
from Source.crops import generate_random_crops, crop_img
from Source.segmentation import run_cellpose_cli, extract_grads_gray, tracking_centroids
from Source.utils import save_mask_as_tif
from Source.finetune import finetune_cellpose, split_dataset
#from main import unstack_images
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit, QSpinBox,
    QComboBox, QCheckBox, QPushButton, QFileDialog, QDoubleSpinBox, QScrollArea
)

# Helper: path selector with label + browse button
def make_path_selector(label_text, placeholder=""):
    layout = QVBoxLayout()
    label = QLabel(label_text)
    path_edit = QLineEdit()
    path_edit.setPlaceholderText(placeholder)
    browse_button = QPushButton("Parcourir")

    def on_browse():
        folder = QFileDialog.getExistingDirectory()
        if folder:
            path_edit.setText(folder)

    browse_button.clicked.connect(on_browse)
    h_layout = QHBoxLayout()
    h_layout.addWidget(path_edit)
    h_layout.addWidget(browse_button)

    layout.addWidget(label)
    layout.addLayout(h_layout)
    return layout, path_edit


# ------------------------- Section Crop ------------------------- #

def add_unstack_section(viewer):
    unstack_group = QGroupBox("Unstack")
    unstack_layout = QVBoxLayout()

    input_layout, input_edit = make_path_selector("Dossier d'entrée :", "Contient le fichier .tif à déstacker")
    output_layout, output_edit = make_path_selector("Dossier de sortie :", "Où sauvegarder les images déstackées")

    unstack_button = QPushButton("Unstack")
    unstack_button.clicked.connect(lambda: on_unstack_clicked(
        input_folder=input_edit.text(),
        output_folder=output_edit.text(),
        viewer=viewer
    ))

    unstack_layout.addLayout(input_layout)
    unstack_layout.addLayout(output_layout)
    unstack_layout.addWidget(unstack_button)

    unstack_group.setLayout(unstack_layout)
    return unstack_group

# ------------------------- Section Crop ------------------------- #
def add_crop_section(viewer):
    crop_group = QGroupBox("Crop")
    crop_layout = QVBoxLayout()

    crop_input_layout, input_edit = make_path_selector("Chemin d'entrée :", "Dossier d'images à croper")
    crop_output_layout, output_edit = make_path_selector("Chemin de sortie :", "Où sauvegarder les crops")

    crop_nfile = QSpinBox()
    crop_nfile.setMinimum(1)
    crop_nfile.setMaximum(1000)
    crop_nfile.setValue(10)

    crop_size = QSpinBox()
    crop_size.setMinimum(32)
    crop_size.setMaximum(2048)
    crop_size.setValue(512)

    crop_pattern = QLineEdit()
    crop_pattern.setPlaceholderText("Ex : _BF_ ou 488")

    crop_button = QPushButton("Crop")
    crop_button.clicked.connect(lambda: on_crop_clicked(
        input_dir=input_edit.text(),
        output_dir=output_edit.text(),
        n_file=crop_nfile.value(),
        size_crop=crop_size.value(),
        pattern=crop_pattern.text(),
        extension=".tif",
        viewer=viewer
    ))

    crop_layout.addLayout(crop_input_layout)
    crop_layout.addLayout(crop_output_layout)
    crop_layout.addWidget(QLabel("Nombre de fichiers à cropper :"))
    crop_layout.addWidget(crop_nfile)
    crop_layout.addWidget(QLabel("Taille des crops :"))
    crop_layout.addWidget(crop_size)
    crop_layout.addWidget(QLabel("Pattern (optionnel) :"))
    crop_layout.addWidget(crop_pattern)
    crop_layout.addWidget(crop_button)

    crop_group.setLayout(crop_layout)
    return crop_group

# ------------------------- Section Segmentation ------------------------- #
def add_segmentation_section(viewer):
    segment_group = QGroupBox("Segmentation")
    segment_layout = QVBoxLayout()

    seg_input_layout, input_edit = make_path_selector("Chemin d'entrée :", "Dossier d'images à segmenter")

    
    model_choice = QComboBox()
    model_choice.addItems(["cpsam", "custom"])

    model_path_layout, model_path_edit = make_path_selector("Chemin du modèle custom :", "Chemin modèle custom (si 'custom')")

    diameter = QSpinBox()
    diameter.setMinimum(0)
    diameter.setMaximum(1000)
    diameter.setValue(0)

    tracking_checkbox = QCheckBox("Activer le tracking après segmentation")

    segment_pattern = QLineEdit()
    segment_pattern.setPlaceholderText("Ex : _00_ ou canalX")

    resize_box = QSpinBox()
    resize_box.setMinimum(64)
    resize_box.setMaximum(2048)
    resize_box.setValue(512)

    # lr_input = QDoubleSpinBox()
    # lr_input.setDecimals(5)
    # lr_input.setSingleStep(0.0001)
    # lr_input.setValue(0.001)
    # lr_input.setMinimum(0.00001)
    # lr_input.setMaximum(1.0)

    segment_button = QPushButton("Segment")
    segment_button.clicked.connect(lambda: on_segment_clicked(
        input_dir=input_edit.text(),
        viewer=viewer,
        model=model_choice.currentText(),
        model_path=model_path_edit.text(),
        enable_tracking=tracking_checkbox.isChecked(),
        pattern=segment_pattern.text(),
        resize=resize_box.value(),
        diameter=diameter.value()
        #lr=lr_input.value()
    ))
    
    segment_layout.addLayout(seg_input_layout)
    segment_layout.addWidget(QLabel("Modèle de segmentation :"))
    segment_layout.addWidget(model_choice)
    segment_layout.addLayout(model_path_layout)
    segment_layout.addWidget(QLabel("Diameter (auto mode is by default 0):"))
    segment_layout.addWidget(diameter)
    segment_layout.addWidget(tracking_checkbox)
    segment_layout.addWidget(QLabel("Pattern :"))
    segment_layout.addWidget(segment_pattern)
    segment_layout.addWidget(QLabel("Resize des images avant segmentation :"))
    segment_layout.addWidget(resize_box)
    #segment_layout.addWidget(QLabel("Learning rate :"))
    #segment_layout.addWidget(lr_input)
    segment_layout.addWidget(segment_button)

    segment_group.setLayout(segment_layout)
    return segment_group

# ------------------------- Section Tracking ------------------------- #
def add_tracking_section(viewer):
    tracking_group = QGroupBox("Tracking")
    tracking_layout = QVBoxLayout()

    tracking_input_layout, tracking_input_edit = make_path_selector("Entrée tracking :", "Dossier à tracker")
    tracking_output_layout, tracking_output_edit = make_path_selector("Sortie tracking :", "Où sauvegarder le résultat")

    tracking_button = QPushButton("Lancer Tracking")
    tracking_button.clicked.connect(lambda: on_tracking_clicked(
        viewer,
        input_dir=tracking_input_edit.text(),
        output_dir=tracking_output_edit.text()
    ))

    tracking_layout.addLayout(tracking_input_layout)
    tracking_layout.addLayout(tracking_output_layout)
    tracking_layout.addWidget(tracking_button)

    tracking_group.setLayout(tracking_layout)
    return tracking_group

# ------------------------- Section Finetune ------------------------- #
def add_finetune_section(viewer):
    finetune_group = QGroupBox("Finetuning")
    finetune_layout = QVBoxLayout()

    save_masks_button = QPushButton("Save corrected masks")
    save_masks_layout, save_masks_edit = make_path_selector("Dossier masques corrigés :", "Où sauvegarder les masques")

    save_masks_button.clicked.connect(lambda: on_save_masks_clicked(
        viewer,
        output_dir=save_masks_edit.text()
    ))

    lr_input = QDoubleSpinBox()
    lr_input.setDecimals(5)
    lr_input.setSingleStep(0.0001)
    lr_input.setValue(0.001)
    lr_input.setMinimum(0.00001)
    lr_input.setMaximum(1.0)

    epochs_input = QSpinBox()
    epochs_input.setMinimum(1)
    epochs_input.setMaximum(1000)
    epochs_input.setValue(100)

    train_data_layout, train_data_edit = make_path_selector("Données pour finetuning :", "Dossier des images + masques")
    model_name = QLineEdit()
    model_name.setPlaceholderText("Nom du modèle en sortie (ex : my_custom_model)")

    finetune_button = QPushButton("Lancer le finetuning")
    finetune_button.clicked.connect(lambda: on_finetune_clicked(
        viewer=viewer,
        train_data_dir=train_data_edit.text(),
        masks_output_dir=save_masks_edit.text(),
        lr=lr_input.value(),
        epochs=epochs_input.value(),
        model_name=model_name.text()
    ))

    finetune_layout.addLayout(save_masks_layout)
    finetune_layout.addWidget(save_masks_button)
    finetune_layout.addWidget(QLabel("Learning rate :"))
    finetune_layout.addWidget(lr_input)
    finetune_layout.addWidget(QLabel("Nombre d'epochs :"))
    finetune_layout.addWidget(epochs_input)
    finetune_layout.addLayout(train_data_layout)
    finetune_layout.addWidget(QLabel("Nom du modèle à sauvegarder :"))
    finetune_layout.addWidget(model_name)
    finetune_layout.addWidget(finetune_button)

    finetune_group.setLayout(finetune_layout)
    return finetune_group

# ------------------------- Assemble UI ------------------------- #
# ------------------------- Assemble UI ------------------------- #
def add_custom_ui_sections(viewer):
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)

    content_widget = QWidget()
    layout = QVBoxLayout()

    layout.addWidget(add_crop_section(viewer))
    layout.addWidget(add_segmentation_section(viewer))
    layout.addWidget(add_tracking_section(viewer))
    layout.addWidget(add_finetune_section(viewer))

    content_widget.setLayout(layout)
    scroll.setWidget(content_widget)

    viewer.window.add_dock_widget(scroll, area='right', name="Pipeline Microsam")


# ------------------------- Appel des fonctions coeur ------------------------- #

def on_crop_clicked(input_dir, output_dir, n_file, size_crop, pattern, extension, viewer):
    if not input_dir or not os.path.isdir(input_dir):
        print("Dossier d'entrée invalide.")
        return
    if not output_dir or not os.path.isdir(output_dir):
        print("Dossier de sortie invalide.")
        return

    patterns = [pattern] if pattern else []
    try:
        rfiles = generate_random_crops(input_dir, n_file, patterns, extension)
        crop_img(rfiles, output_dir, size_crop)
        print(f"{len(rfiles)} images croppées et sauvegardées.")
    except Exception as e:
        print(f"Erreur pendant le crop : {e}")

# def on_unstack_clicked(input_folder, output_folder, viewer):
#     unstack_images(input_folder=input_folder, output_folder=output_folder)

def on_segment_clicked(input_dir, viewer,model,model_path,enable_tracking,pattern,resize, diameter):
    run_cellpose_cli(input_folder=input_dir, model_type=model, custom_model=model_path, diameter=diameter)
    extract_grads_gray(input_folder=input_dir)
    #TODO: go through the segmented and grads file and stack them to open them in napari

def on_save_masks_clicked(viewer, output_dir):
    save_mask_as_tif(viewer=viewer, output_dir=output_dir)

def on_finetune_clicked(viewer, train_data_dir, masks_output_dir, lr, epochs, model_name):
    split_dataset(finetune_dir=masks_output_dir)
    finetune_cellpose(output_path=masks_output_dir, epochs=epochs, lr=lr, model_name=model_name)

def on_tracking_clicked(viewer, input_dir, output_dir):
    tracking_centroids(input_folder=input_dir)
    # Take as input the stack we segmented, but user has to provide it