# 04. pyqt_key_event.py
# PyQt 키 이벤트
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        # 창 크기 고정
        self.setFixedSize(200, 200)
        # 창 제목 설정
        self.setWindowTitle('GA-Mario')

        self.keyLabel = QLabel(self)
        self.keyLabel.setGeometry(0, 0, 200, 200)

        # 창 띄우기
        self.show()

    # 키를 누를 때
    def keyPressEvent(self, event):
        key = event.key()
        self.keyLabel.setText('press ' + str(key))

    # 키를 땔 때
    def keyReleaseEvent(self, event):
        key = event.key()
        self.keyLabel.setText('release ' + str(key))


if __name__ == '__main__':
   app = QApplication(sys.argv)
   window = MyApp()
   sys.exit(app.exec_())
