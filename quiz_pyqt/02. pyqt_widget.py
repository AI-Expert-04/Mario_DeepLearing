import sys
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton
import numpy as np


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(300, 300)
        self.setWindowTitle('GA Mario')

        self.label_image = QLabel(self)
        image = np.array(
            [
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ],
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ],
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ]
            ]
        )
        print(image)
        original = QImage(image, image.shape[1], image.shape[0], QImage.Format_RGB888)
        qimage = QImage(original)
        pixmap = QPixmap(qimage)
        pixmap = pixmap.scaled(100, 100, Qt.IgnoreAspectRatio)
        self.label_image.setPixmap(pixmap)
        self.label_image.setGeometry(0, 0, 100, 100)

        self.label_text = QLabel(self)
        self.label_text.setText('가나다')
        self.label_text.setGeometry(100, 100, 100, 100)

        self.button = QPushButton(self)
        self.button.setText('버튼')
        self.button.setGeometry(200, 200, 100, 100)

        self.show()


if __name__ == '__main__':
   app = QApplication(sys.argv)
   window = MyApp()
   sys.exit(app.exec_())
