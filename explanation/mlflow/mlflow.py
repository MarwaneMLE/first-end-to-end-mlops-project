import mlflow


def calulator (a, b, operation=None):
    if operation == "add":
        return a + b
    if operation == "sub":
        return a - b
    if operation == "mul":
        return a * b
    if operation == "div":
        return a / b
 

if __name__ == "__main__":
    a, b, opt = 11, 20, "add"
    with mlflow.start_run():
        res = calulator(a, b, opt)
        mlflow.log_param("a", a)
        mlflow.log_param("b", b)
        mlflow.log_param("opt", opt)

        print(f"Result is: {res}")

        mlflow.log_param("res", res)
