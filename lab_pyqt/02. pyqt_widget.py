# 02. pyqt_widget.py
# PyQt 위젯
import sys
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication,\
    QWidget, QLabel, QPushButton
import numpy as np


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        # 창 크기 고정
        self.setFixedSize(400, 300)
        # 창 제목 설정
        self.setWindowTitle('GA Mario')

        # 버튼
        button = QPushButton(self)
        button.setText('버튼')
        button.setGeometry(100, 100, 50, 50)

        # 텍스트
        label_text = QLabel(self)
        label_text.setText('가나다')
        label_text.setGeometry(200, 150, 50, 100)

        # 이미지
        label_image = QLabel(self)

        image = np.array([[[255, 255, 255], [255, 255, 255]], [[255, 255, 255], [255, 255, 255]]])
        qimage = QImage(image, image.shape[1], image.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap(qimage)
        pixmap = pixmap.scaled(100, 100, Qt.IgnoreAspectRatio)

        label_image.setPixmap(pixmap)
        label_image.setGeometry(0, 0, 100, 100)

        self.show()


if __name__ == '__main__':
   app = QApplication(sys.argv)
   window = MyApp()
   sys.exit(app.exec_())
