import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

def parse_xml(xml_file):
    """Parse the XML file to extract bounding box coordinates."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    boxes = []

    for object in root.findall('object'):
        bndbox = object.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        boxes.append((xmin, ymin, xmax, ymax))

    return boxes

def draw_boxes(image_path, boxes, output_path):
    """Draw bounding boxes on the image."""
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw each box
    for (xmin, ymin, xmax, ymax) in boxes:
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)  # Blue boxes with thickness of 2

    # Save or display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(output_path)
    plt.show()

# Paths to the XML file and image
xml_file = 'Annotations/levle3_110.xml'
image_path = 'ImageSets/JPEGImages/levle3_110.jpg'
output_path = 'image_with_boxes.jpg'

# Parse XML and draw boxes
boxes = parse_xml(xml_file)
draw_boxes(image_path, boxes, output_path)
