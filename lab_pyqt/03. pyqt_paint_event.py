# 03. pyqt_paint_event.py
# PyQt 그리기
import sys
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        # 창 크기 고정
        self.setFixedSize(200, 300)
        # 창 제목 설정
        self.setWindowTitle('MyApp')
        # 창 띄우기
        self.show()

    # 창이 업데이트 될 때마다 실행되는 함수
    def paintEvent(self, event):
        # 그리기 도구
        painter = QPainter()
        # 그리기 시작
        painter.begin(self)

        # 펜 설정 (테두리)
        painter.setPen(QPen(Qt.blue, 2.0, Qt.SolidLine))
        # 선 그리기
        painter.drawLine(0, 10, 200, 100)

        # RGB 색상으로 펜 설정
        painter.setPen(QPen(QColor.fromRgb(255, 0, 0), 3.0, Qt.SolidLine))
        # 브러쉬 설정 (채우기)
        painter.setBrush(QBrush(Qt.blue))
        # 직사각형 그리기
        painter.drawRect(0, 100, 100, 100)

        painter.setPen(QPen(Qt.black, 1.0, Qt.SolidLine))
        # RGB 색상으로 브러쉬 설정
        painter.setBrush(QBrush(QColor.fromRgb(0, 255, 0)))
        # 타원 그리기
        painter.drawEllipse(100, 100, 100, 100)

        painter.setPen(QPen(Qt.cyan, 1.0, Qt.SolidLine))
        # 브러쉬 초기화
        painter.setBrush(Qt.NoBrush)
        # 텍스트 그리기
        painter.drawText(0, 250, 'abcd')

        # 그리기 끝
        painter.end()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyApp()
    sys.exit(app.exec_())

