import torch


class LinearRegression:
    def __init__(self):
        """
        初始化线性回归模型。
        """
        self.weight = None  # 系数 (w1, w2, ...)，将是 torch.Tensor
        self.bias = None  # 偏置 (b)，将是 torch.Tensor

    def fit(self, X, y):
        """
        使用普通最小二乘法拟合线性模型。

        参数:
        ----------
        X : array-like, shape (n_samples, n_features)
            训练数据。
        y : array-like, shape (n_samples,)
            目标值。
        """
        # 确保输入是 PyTorch 张量，并指定数据类型为 float32
        # PyTorch 的矩阵运算通常在 float 类型上进行
        X = torch.as_tensor(X, dtype=torch.float64)
        y = torch.as_tensor(y, dtype=torch.float64)

        # 如果 X 是一维的，将其转换为 (1, n_features) 的二维张量
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # 在 X 的最后一列添加全为 1 的列，以计算偏置 b
        # [X, 1] @ [w, b]^T = X @ w + b
        ones_column = torch.ones(X.shape[0], 1, dtype=X.dtype)
        X_b = torch.cat([X, ones_column], dim=1)

        # 使用 torch.linalg.lstsq 求解线性方程组 X_b @ coeffs = y
        # 这是普通最小二乘法的解析解
        solution = torch.linalg.lstsq(X_b, y)
        coeffs = solution.solution

        # 从解中分离出权重和偏置
        self.weight = coeffs[:-1]  # 系数
        self.bias = coeffs[-1]  # 截距（偏置）

        return self

    def predict(self, X):
        """
        使用训练好的模型进行预测。

        参数:
        ----------
        X : array-like, shape (n_samples, n_features)
            待预测的数据。

        返回:
        -------
        y_pred : torch.Tensor, shape (n_samples,)
            预测结果。
        """
        if self.weight is None or self.bias is None:
            raise RuntimeError("模型尚未训练，请先调用 fit() 方法。")

        # 确保输入是 PyTorch 张量
        X = torch.as_tensor(X, dtype=torch.float64)

        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X @ self.weight + self.bias

    def serialize(self):
        """将模型参数序列化为字典。"""
        # 直接返回张量，因为 torch.save 可以处理它们
        return {
            "weight": self.weight,
            "bias": self.bias,
        }

    @classmethod
    def deserialize(cls, data):
        """从字典加载模型参数。"""
        model = cls()
        model.weight = data["weight"]
        model.bias = data["bias"]
        return model

    def save_model(self, model_path):
        # 使用 torch.save 保存包含参数的字典
        torch.save(self.serialize(), model_path)

    @classmethod
    def load_model(cls, model_path):
        # 使用 torch.load 加载数据，并反序列化
        data = torch.load(model_path)
        return cls.deserialize(data)


def r2_score(y_true, y_pred):
    """
    计算 R-squared (R²) 决定系数。

    参数:
    ----------
    y_true : array-like, shape (n_samples,)
        真实目标值。
    y_pred : array-like, shape (n_samples,)
        模型预测的目标值。

    返回:
    -------
    score : float
        R² 分数。
    """
    # 确保输入是 PyTorch 张量
    y_true = torch.as_tensor(y_true)
    y_pred = torch.as_tensor(y_pred)

    # 计算总平方和 (SS_tot)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)

    # 计算残差平方和 (SS_res)
    ss_res = torch.sum((y_true - y_pred) ** 2)

    # 处理 ss_tot 为 0 的特殊情况
    if ss_tot == 0:
        # 如果残差也为0，说明完美预测，返回1.0
        # 否则，模型无意义，返回0.0
        return 1.0 if ss_res == 0 else 0.0

    # 计算 R²
    r2 = 1 - (ss_res / ss_tot)
    return r2.item()  # 返回一个标准的 Python float
