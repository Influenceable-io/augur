mod common;

use augur::channel::Channel;
use augur::types::{ActionResult, ActionType};
use serde_json::json;
use std::sync::Arc;

#[tokio::test]
async fn test_channel_roundtrip() {
    let channel = Arc::new(Channel::new());

    let agent_id: i64 = 42;
    let message = json!({"content": "Hello, world!"});
    let action = ActionType::CreatePost;

    // Agent writes a request.
    let msg_id = channel
        .write_to_receive_queue(agent_id, message.clone(), action)
        .await;

    // Platform receives the request.
    let (recv_msg_id, (recv_agent_id, recv_message, recv_action)) =
        channel.receive_from().await;

    assert_eq!(recv_msg_id, msg_id);
    assert_eq!(recv_agent_id, agent_id);
    assert_eq!(recv_message, message);
    assert_eq!(recv_action, action);

    // Platform sends a response.
    let result = ActionResult::ok(json!({"post_id": 1}));
    channel
        .send_to(msg_id, agent_id, result.clone())
        .await;

    // Agent reads the response.
    let (resp_msg_id, resp_agent_id, resp_result) =
        channel.read_from_send_queue(msg_id).await;

    assert_eq!(resp_msg_id, msg_id);
    assert_eq!(resp_agent_id, agent_id);
    assert!(resp_result.success);
    assert_eq!(resp_result.data, json!({"post_id": 1}));
}

#[tokio::test]
async fn test_channel_concurrent_requests() {
    let channel = Arc::new(Channel::new());
    let num_requests: usize = 10;

    // Spawn agent tasks that each write a request and wait for a response.
    let mut agent_handles = Vec::new();

    for i in 0..num_requests {
        let agent_id = i as i64;
        let ch = Arc::clone(&channel);
        let message = json!({"index": i});

        let handle = tokio::spawn(async move {
            let msg_id = ch
                .write_to_receive_queue(agent_id, message, ActionType::CreatePost)
                .await;

            let (resp_msg_id, resp_agent_id, resp_result) =
                ch.read_from_send_queue(msg_id).await;

            assert_eq!(resp_msg_id, msg_id);
            assert_eq!(resp_agent_id, agent_id);
            assert!(resp_result.success);
            let resp_index = resp_result.data["index"].as_u64().unwrap() as usize;
            assert_eq!(resp_index, i);

            msg_id
        });

        agent_handles.push(handle);
    }

    // Platform task: receive all requests and respond.
    let platform_ch = Arc::clone(&channel);
    let platform_handle = tokio::spawn(async move {
        for _ in 0..num_requests {
            let (msg_id, (agent_id, message, _action)) =
                platform_ch.receive_from().await;

            let index = message["index"].as_u64().unwrap();
            let result = ActionResult::ok(json!({"index": index}));
            platform_ch.send_to(msg_id, agent_id, result).await;
        }
    });

    // Wait for all tasks.
    platform_handle.await.expect("platform task panicked");
    for handle in agent_handles {
        handle.await.expect("agent task panicked");
    }
}
