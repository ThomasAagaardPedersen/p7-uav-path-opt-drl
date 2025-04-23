import airsim
from datetime import datetime

client = airsim.VehicleClient()
client.confirmConnection()
framecounter = 1

prevtimestamp = datetime.now()

while(framecounter <= 500):
    if framecounter%150 == 0:
        client.simGetImages([airsim.ImageRequest("high_res", airsim.ImageType.Scene, False, False)])
        print("High resolution image captured.")

    if framecounter%30 == 0:
        now = datetime.now()
        print(f"Time spent for 30 frames: {now-prevtimestamp}")
        prevtimestamp = now

    client.simGetImages([airsim.ImageRequest("low_res", airsim.ImageType.Scene, False, False)])
    framecounter += 1