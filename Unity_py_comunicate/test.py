from zmq_modules.zmq_unity_pull_target import get_target_generator
import sys

def test_get_target_generator():
    print("Starting target generator test. Waiting for messages on tcp://*:6005...")
    gen = get_target_generator()
    try:
        for msg in gen:
            print("Received message:", msg)
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(0)

if __name__ == "__main__":
    test_get_target_generator()