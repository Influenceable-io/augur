use dashmap::DashMap;
use serde_json::Value;
use tokio::sync::{mpsc, Mutex, Notify};
use uuid::Uuid;

use crate::types::{ActionResult, ActionType};

/// Request payload sent from agent to platform.
pub type ChannelRequest = (Uuid, (i64, Value, ActionType));

/// Response payload sent from platform to agent.
pub type ChannelResponse = (Uuid, i64, ActionResult);

/// Async message-passing channel between agents and the platform.
///
/// Mirrors OASIS's Channel class:
/// - Agents write requests via `write_to_receive_queue`
/// - Platform reads requests via `receive_from`
/// - Platform writes responses via `send_to`
/// - Agents read responses via `read_from_send_queue`
pub struct Channel {
    receive_tx: mpsc::UnboundedSender<ChannelRequest>,
    receive_rx: Mutex<mpsc::UnboundedReceiver<ChannelRequest>>,
    send_dict: DashMap<Uuid, ChannelResponse>,
    notify: Notify,
}

impl Channel {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        Self {
            receive_tx: tx,
            receive_rx: Mutex::new(rx),
            send_dict: DashMap::new(),
            notify: Notify::new(),
        }
    }

    /// Platform receives the next request from any agent.
    pub async fn receive_from(&self) -> ChannelRequest {
        let mut rx = self.receive_rx.lock().await;
        rx.recv().await.expect("Channel receive_tx dropped unexpectedly")
    }

    /// Platform sends a response back to an agent.
    pub async fn send_to(&self, message_id: Uuid, agent_id: i64, result: ActionResult) {
        self.send_dict.insert(message_id, (message_id, agent_id, result));
        self.notify.notify_waiters();
    }

    /// Agent writes a request and receives a message_id to poll for the response.
    pub async fn write_to_receive_queue(
        &self,
        agent_id: i64,
        message: Value,
        action: ActionType,
    ) -> Uuid {
        let message_id = Uuid::new_v4();
        self.receive_tx
            .send((message_id, (agent_id, message, action)))
            .expect("Channel receive_rx dropped unexpectedly");
        message_id
    }

    /// Agent polls for a response by message_id. Blocks until available.
    pub async fn read_from_send_queue(&self, message_id: Uuid) -> ChannelResponse {
        loop {
            if let Some((_, response)) = self.send_dict.remove(&message_id) {
                return response;
            }
            self.notify.notified().await;
        }
    }
}

impl Default for Channel {
    fn default() -> Self {
        Self::new()
    }
}
