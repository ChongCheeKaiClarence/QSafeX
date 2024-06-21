from gradio_client import Client, handle_file

inputImage = 'cropped_images\crop_0_0.jpg'
inputTask = 'Region to Category'
inputText = 'shoes'

client = Client("SixOpen/Florence-2-large-ft")
result = client.predict(
		image=handle_file(inputImage),
		task=inputTask,
		text=inputText,
		api_name="/process_image"
)
print(result[0])