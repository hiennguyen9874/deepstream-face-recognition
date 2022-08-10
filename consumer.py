import argparse
import json
import os

# import uuid
from kafka import KafkaConsumer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--topic",
    type=str,
    default="test",
)
args = parser.parse_args()


consumer = KafkaConsumer(
    args.topic,
    bootstrap_servers="10.1.1.41:9092",
    auto_offset_reset="latest",
    enable_auto_commit=True,
    value_deserializer=lambda x: json.loads(x.decode("utf-8").replace("'", '"')),
)

# do a dummy poll to retrieve some message
consumer.poll()

# go to end of the stream
# consumer.seek_to_end()

if os.path.exists("./outputs/jsons/result.txt"):
    os.remove("./outputs/jsons/result.txt")

for event in consumer:
    # data = json.loads(event.value.decode("utf-8"))
    # # print(f"source_id: {event_data['source_id']}, frame: {event_data['id']}, buf_pts: {event_data['buf_pts']}") # noqa: B950
    event_data = event.value
    print(event_data)
    # print(
    #     f"frame: {event_data['frame_number']}, source_id: {event_data['source_id']}, num_frames_in_batch: {event_data['num_frames_in_batch']}, max_frames_in_batch: {event_data['max_frames_in_batch']}, ntp_timestamp: {event_data['ntp_timestamp']}, msg_timestamp: {event_data['msg_timestamp']}"  # noqa: B950
    # )
    # print(list(event_data.keys()))

    with open("./outputs/jsons/result.txt", "a") as f:
        f.write(json.dumps(event_data) + "\n")
