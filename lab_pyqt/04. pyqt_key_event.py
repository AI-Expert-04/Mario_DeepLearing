# 04. pyqt_key_event.py
# PyQt 키 이벤트
import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        # 창 크기 고정
        self.setFixedSize(400, 300)
        # 창 제목 설정
        self.setWindowTitle('MyApp')
        # 창 띄우기
        self.show()

    # 키를 누를 때
    def keyPressEvent(self, event):
        key = event.key()
        print(str(key) + ' press')

    # 키를 뗄 때
    def keyReleaseEvent(self, event):
        key = event.key()
        print(str(key) + ' release')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyApp()
    sys.exit(app.exec_())
