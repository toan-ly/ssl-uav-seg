uavid:
	uv run gdown --folder https://drive.google.com/drive/u/3/folders/1JKuMrJzKY95FIMuDJM23qx3-AeOOH5jq -O data/
	unzip data/UAV/uav_data.zip -d data/
	rm data/UAV/uav_data.zip
# 	mv data/archive data/UAVid
	rm -rf data/UAV
	rm -rf data/__MACOSX