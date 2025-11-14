dataset:
	uv run gdown --folder https://drive.google.com/drive/u/3/folders/1G9d5yPcZyWirjg5wXILNthetzI_30z6A -O data/
	unzip data/UAV/uav_dataset.zip -d data/
	rm data/UAV/uav_dataset.zip
	mv data/archive data/UAVid
	rm -rf data/UAV
	rm -rf data/__MACOSX