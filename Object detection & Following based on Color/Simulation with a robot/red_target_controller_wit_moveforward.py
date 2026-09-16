from controller import Robot
import numpy as np
import cv2

# Create robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Camera setup
camera = robot.getDevice("camera")
camera.enable(timestep)

# Motor setup
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))

left_motor.setVelocity(0)
right_motor.setVelocity(0)

# Controller parameters
Kp = 0.01

forward_speed = 2.0
max_turn_speed = 3.0
max_wheel_speed = 6.28

# Number of red pixels at which robot stops
STOP_THRESHOLD = 1000


# OpenCV window
cv2.namedWindow(
    "E-puck Camera",
    cv2.WINDOW_NORMAL
)

# Main loop
while robot.step(timestep) != -1:
    # Get camera image

    image = camera.getImage()

    image = np.frombuffer(
        image,
        dtype=np.uint8
    )

    image = image.reshape(
        (
            camera.getHeight(),
            camera.getWidth(),
            4
        )
    )

    # Webots BGRA → OpenCV BGR
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGRA2BGR
    )

    # Convert BGR → HSV

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # Red color ranges

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([179, 255, 255])


    # Create red mask
    mask1 = cv2.inRange(
        hsv,
        lower_red1,
        upper_red1
    )

    mask2 = cv2.inRange(
        hsv,
        lower_red2,
        upper_red2
    )

    mask = mask1 | mask2


    # Find red pixels

    y, x = np.where(mask > 0)

    red_pixel_count = len(x)

    # Check whether red object exists

    if red_pixel_count > 100:

        # Find red object's center-

        target_x = int(np.mean(x))
        target_y = int(np.mean(y))

        # Camera image center

        image_center_x = camera.getWidth() // 2

        # Calculate horizontal error

        error = target_x - image_center_x

        # STOP CONDITION

        if red_pixel_count >= STOP_THRESHOLD:

            # Stop robot
            left_motor.setVelocity(0)
            right_motor.setVelocity(0)

            print("================================")
            print("TARGET REACHED!")
            print("Red pixels:", red_pixel_count)
            print("Robot stopped.")
            print("================================")


        else:

            # Proportional turning

            turn = Kp * error

            turn = np.clip(
                turn,
                -max_turn_speed,
                max_turn_speed
            )


            # Forward + turning

            left_speed = forward_speed + turn
            right_speed = forward_speed - turn


            # Limit wheel speeds

            left_speed = np.clip(
                left_speed,
                -max_wheel_speed,
                max_wheel_speed
            )

            right_speed = np.clip(
                right_speed,
                -max_wheel_speed,
                max_wheel_speed
            )


            # Send speeds to motors

            left_motor.setVelocity(
                left_speed
            )

            right_motor.setVelocity(
                right_speed
            )


            # Print controller information

            print(
                "Target X:", target_x,
                "| Error:", error,
                "| Red pixels:", red_pixel_count,
                "| Left:", round(left_speed, 2),
                "| Right:", round(right_speed, 2)
            )

        # Draw target center

        cv2.circle(
            image,
            (target_x, target_y),
            5,
            (0, 255, 0),
            -1
        )


        # Draw camera center

        cv2.line(
            image,
            (
                image_center_x,
                0
            ),
            (
                image_center_x,
                camera.getHeight()
            ),
            (255, 0, 0),
            1
        )


        # Display red pixel count

        cv2.putText(
            image,
            f"Red pixels: {red_pixel_count}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Red object NOT detected

    else:

        left_motor.setVelocity(0)
        right_motor.setVelocity(0)

        print("Red object not detected")

    # Display camera

    display = cv2.resize(
        image,
        (520, 390)
    )

    cv2.imshow(
        "E-puck Camera",
        display
    )

    cv2.waitKey(1)

# Stop motors when simulation ends

left_motor.setVelocity(0)
right_motor.setVelocity(0)

cv2.destroyAllWindows()