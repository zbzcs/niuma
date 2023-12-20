package com.itniuma.niuma.R;

public class Response<T> {
    private int code;       // 状态码
    private String message; // 提示信息
    private T data;         // 具体的数据

    // 构造函数
    public Response(int code, String message, T data) {
        this.code = code;
        this.message = message;
        this.data = data;
    }

    // Getter 和 Setter 略

    // 静态方法，方便创建成功的响应
    public static <T> Response<T> success(T data) {
        return new Response<>(200, "OK", data);
    }

    // 静态方法，方便创建失败的响应
    public static <T> Response<T> error(int code, String message) {
        return new Response<>(code, message, null);
    }
}
