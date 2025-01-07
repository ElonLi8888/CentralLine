import cv2
import numpy as np


def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)


def canny_edge_detection(image, low_threshold=150, high_threshold=200):
    return cv2.Canny(image, low_threshold, high_threshold)


def dilate_with_buffer(image, buffer_radius=5):
    kernel = np.ones((buffer_radius, buffer_radius), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


def detect_centerline(image, orientation="vertical", buffer_radius=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = apply_gaussian_blur(gray)
    edges = canny_edge_detection(blurred)
    dilated_edges = dilate_with_buffer(edges, buffer_radius)

    line_image = image.copy()
    center_pos = None
    lines = cv2.HoughLinesP(dilated_edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    if lines is not None:
        relevant_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if orientation == "vertical" and abs(x2 - x1) < abs(y2 - y1):
                relevant_lines.append(((x1 + x2) // 2, abs(y2 - y1)))
            elif orientation == "horizontal" and abs(y2 - y1) < abs(x2 - x1):
                relevant_lines.append(((y1 + y2) // 2, abs(x2 - x1)))

        if relevant_lines:
            if orientation == "vertical":
                weighted_sum = sum(x * length for x, length in relevant_lines)
                total_length = sum(length for _, length in relevant_lines)
                center_x = int(weighted_sum / total_length) if total_length > 0 else image.shape[1] // 2
                cv2.line(line_image, (center_x, 0), (center_x, line_image.shape[0]), (0, 255, 0), 2)
                center_pos = center_x
            elif orientation == "horizontal":
                weighted_sum = sum(y * length for y, length in relevant_lines)
                total_length = sum(length for _, length in relevant_lines)
                center_y = int(weighted_sum / total_length) if total_length > 0 else image.shape[0] // 2
                cv2.line(line_image, (0, center_y), (line_image.shape[1], center_y), (0, 0, 255), 2)
                center_pos = center_y

    return line_image, lines, center_pos


def resize_frame(frame):
    return cv2.resize(frame, (1280, 720))


def detect_lines_orientation(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = canny_edge_detection(gray)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    vertical_lines = 0
    horizontal_lines = 0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < abs(y2 - y1):
                vertical_lines += 1
            elif abs(y2 - y1) < abs(x2 - x1):
                horizontal_lines += 1

    return "vertical" if vertical_lines > horizontal_lines else "horizontal"


def draw_parallel_lines(image, orientation="vertical"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = apply_gaussian_blur(gray)
    edges = canny_edge_detection(blurred)
    dilated_edges = dilate_with_buffer(edges)

    line_image = image.copy()
    lines = None

    lines = cv2.HoughLinesP(dilated_edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if orientation == "vertical" and abs(x2 - x1) < abs(y2 - y1):
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            elif orientation == "horizontal" and abs(y2 - y1) < abs(x2 - x1):
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

    return line_image, lines


def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = resize_frame(frame)


        orientation = detect_lines_orientation(frame_resized)

        if orientation == "horizontal":
            processed_frame, lines = draw_parallel_lines(frame_resized, orientation="horizontal")
            processed_frame, lines, center_pos = detect_centerline(processed_frame, orientation="horizontal")

            cv2.putText(processed_frame, "Orientation: Horizontal", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if center_pos:
                cv2.putText(processed_frame, f"Center Y: {center_pos}", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            processed_frame, lines = draw_parallel_lines(frame_resized, orientation="vertical")
            processed_frame, lines, center_pos = detect_centerline(processed_frame, orientation="vertical")

            cv2.putText(processed_frame, "Orientation: Vertical", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 100), 2)
            if center_pos:
                cv2.putText(processed_frame, f"Center X: {center_pos}", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 100), 2)


        if lines is not None:
            cv2.putText(processed_frame, f"Lines detected: {len(lines)}", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.imshow('Line Detection', processed_frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
