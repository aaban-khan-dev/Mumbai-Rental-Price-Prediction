import os

def error_message_detail(error,error_detail):
    _,_,exc_tb = error_detail.exc_info()
    if exc_tb is None:
        return str(error)
    file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    line_number = exc_tb.tb_lineno
    return f"Error occurred in python script name [{file_name}] line number [{line_number}] error message [{str(error)}]"

class RentException(Exception):
    def __init__(self,error_message,error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message