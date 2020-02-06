

class User:
    name = None
    position = None
    checkinTime = None
    checkoutTime = None
    checkinFace = None
    checkoutFace = None
    emotion = None
    namePulse = None
    workNeedDone = None

    def __init__(self, namePulse, name, position, checkinTime = None, checkoutTime = None, checkinFace = None, checkoutFace = None, emotion = None, workNeedDone = ""):
        self.namePulse = namePulse
        self.name = name
        self.position = position
        self.checkinTime = checkinTime
        self.checkoutTime = checkoutTime
        self.checkinFace = checkinFace
        self.checkoutFace = checkoutFace
        self.emotion = emotion
        self.workNeedDone = workNeedDone

    def getMessage(self):
        msg = "Xin chào " + self.namePulse +" " + (self.name) + ", chúc anh một buổi sáng tốt lành! "
        msg += self.namePulse + " còn công việc " + self.workNeedDone + " cần hoàn thành ngay. chúc anh sớm hoàn thành công việc"
        return msg

    def getFakeData():
        users = {}
        users["tien"] = User("Anh", "Đoàn Cảnh Tiền", "Mobile Developer",workNeedDone="release version mới")
        users["manh"] =  User("Anh", "Dương Văn Mạnh", "CRM Developer",workNeedDone="Fix bug màn hình import lead")
        users["ngoc"] =  User("Anh", "Tuấn Ngọc", "AI Intern Developer",workNeedDone="Hoàn thành hàm nhận diện khuôn mặt")
        users["tuananh"] =  User("Anh", "Lê Tuấn Anh", "CRM Developer",workNeedDone="Fix bug trí nam")
        users["sang"] =  User("Anh", "Sang", "CRM Developer",workNeedDone="Fix bug crm")

        return users