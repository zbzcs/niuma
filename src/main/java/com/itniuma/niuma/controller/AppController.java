package com.itniuma.niuma.controller;


import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.itniuma.niuma.R.Response;
import com.itniuma.niuma.utils.YourDataClass;
import com.tencentcloudapi.common.Credential;
import com.tencentcloudapi.common.SSEResponseModel;
import com.tencentcloudapi.common.exception.TencentCloudSDKException;
import com.tencentcloudapi.common.profile.ClientProfile;
import com.tencentcloudapi.hunyuan.v20230901.HunyuanClient;
import com.tencentcloudapi.hunyuan.v20230901.models.ChatStdRequest;
import com.tencentcloudapi.hunyuan.v20230901.models.ChatStdResponse;
import com.tencentcloudapi.hunyuan.v20230901.models.Message;

import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import lombok.extern.slf4j.Slf4j;

import org.springframework.web.bind.annotation.*;

import java.util.HashMap;


@RestController
@Slf4j
@Api("你好")
@RequestMapping("/api")
public class AppController {
    @ApiOperation("文案生成接口")
    @PostMapping("/write")

    public String write(@RequestBody String sbmsg) {

        System.out.println(sbmsg);
        StringBuilder result = new StringBuilder("");
        try {
            Credential cred = new Credential("AKIDFmGpSxx0vmV6iSGNH3sf3TjokXuBFMRV", "88ziP2lPsFaov6HZt3pYNmCg0HasHn8F");

            ClientProfile clientProfile = new ClientProfile();
            HunyuanClient client = new HunyuanClient(cred, "ap-guangzhou", clientProfile);

            ChatStdRequest req = new ChatStdRequest();
            Message msg = new Message();
            msg.setRole("user");
            HashMap<String, String> hashMap = JSON.parseObject(sbmsg, HashMap.class);
            String s = hashMap.get("msg");
            msg.setContent(s);
            req.setMessages(new Message[]{
                    msg
            });

            ChatStdResponse resp = client.ChatStd(req);

            for (SSEResponseModel.SSE e : resp) {
                String jsonString=e.Data;
                YourDataClass yourData = JSON.parseObject(jsonString, YourDataClass.class);
                result.append(yourData.getChoices().get(0).getDelta().getContent());
            }
        } catch (TencentCloudSDKException e) {
            System.out.println(e);
        }
        System.out.println(result.toString().trim());
        return result.toString().trim(); // 返回所有消息的字符串
    }

    @ApiOperation("图像生成接口")
    @PostMapping("/draw")
    public Response draw() {
        return Response.success(null);
    }
}
