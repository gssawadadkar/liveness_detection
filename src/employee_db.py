from flask_mysqldb import MySQL

class EmployeeDB:
    def __init__(self, app=None):
        if app:
            self.init_app(app)

    def init_app(self, app):
        self.mysql = MySQL(app)

    def get_employees(self, app):
        with app.app_context():
            cursor = self.mysql.connection.cursor()
            cursor.execute("SELECT * FROM employees")
            employees = cursor.fetchall()
            cursor.close()
            
            return employees
