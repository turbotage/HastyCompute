//! The value bank: UUID → (value, name, metadata JSON). Readers take an `Arc<Value>` and encode without holding the
//! lock; slice writes copy-on-write (`Arc::make_mut`). Every change is broadcast as an [`Event`] (viewers, logs).

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use hasty_value::{slice, Value};
use tokio::sync::broadcast;
use uuid::Uuid;

use crate::Error;

#[derive(Clone, Debug)]
pub struct Event {
    /// put | update | delete | metadata | show | log
    pub kind: String,
    pub id: Option<Uuid>,
    pub name: String,
    pub message: String,
}

#[derive(Clone)]
pub struct Entry {
    pub name: String,
    pub value: Arc<Value>,
    pub metadata: String,
    pub created_ms: i64,
    pub updated_ms: i64,
}

/// Entry description without the value.
#[derive(Clone, Debug)]
pub struct EntryInfo {
    pub id: Uuid,
    pub name: String,
    pub summary: String,
    pub metadata: String,
    pub nbytes: u64,
    pub created_ms: i64,
    pub updated_ms: i64,
}

pub struct Bank {
    map: RwLock<HashMap<Uuid, Entry>>,
    events: broadcast::Sender<Event>,
}

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

impl Default for Bank {
    fn default() -> Self {
        Self::new()
    }
}

impl Bank {
    pub fn new() -> Self {
        Self { map: RwLock::new(HashMap::new()), events: broadcast::channel(1024).0 }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.events.subscribe()
    }

    pub fn notify(&self, kind: &str, id: Option<Uuid>, name: &str, message: &str) {
        let _ = self.events.send(Event { kind: kind.into(), id, name: name.into(), message: message.into() });
    }

    pub fn put(&self, name: &str, value: Value, metadata: &str) -> Uuid {
        let id = Uuid::new_v4();
        let t = now_ms();
        let name = if name.is_empty() { format!("{} {}", value.kind(), &id.simple().to_string()[..8]) } else { name.to_string() };
        self.map.write().unwrap().insert(id, Entry { name: name.clone(), value: Arc::new(value), metadata: if metadata.is_empty() { "{}".into() } else { metadata.into() }, created_ms: t, updated_ms: t });
        self.notify("put", Some(id), &name, "");
        id
    }

    pub fn get(&self, id: &Uuid) -> Result<Entry, Error> {
        self.map.read().unwrap().get(id).cloned().ok_or(Error::NotFound(*id))
    }

    pub fn value(&self, id: &Uuid) -> Result<Arc<Value>, Error> {
        Ok(self.get(id)?.value)
    }

    /// replace the whole value, or the part selected by `path`
    pub fn write(&self, id: &Uuid, path: &str, value: Value) -> Result<(), Error> {
        let name = {
            let mut m = self.map.write().unwrap();
            let e = m.get_mut(id).ok_or(Error::NotFound(*id))?;
            if path.is_empty() {
                e.value = Arc::new(value);
            } else {
                slice::set(Arc::make_mut(&mut e.value), path, value)?;
            }
            e.updated_ms = now_ms();
            e.name.clone()
        };
        self.notify("update", Some(*id), &name, path);
        Ok(())
    }

    pub fn delete(&self, id: &Uuid) -> Result<(), Error> {
        let e = self.map.write().unwrap().remove(id).ok_or(Error::NotFound(*id))?;
        self.notify("delete", Some(*id), &e.name, "");
        Ok(())
    }

    pub fn set_metadata(&self, id: &Uuid, metadata: &str) -> Result<EntryInfo, Error> {
        if !metadata.is_empty() {
            serde_json::from_str::<serde_json::Value>(metadata).map_err(|e| Error::Invalid(format!("metadata is not JSON: {e}")))?;
        }
        let name = {
            let mut m = self.map.write().unwrap();
            let e = m.get_mut(id).ok_or(Error::NotFound(*id))?;
            e.metadata = if metadata.is_empty() { "{}".into() } else { metadata.into() };
            e.updated_ms = now_ms();
            e.name.clone()
        };
        self.notify("metadata", Some(*id), &name, "");
        self.info(id)
    }

    pub fn info(&self, id: &Uuid) -> Result<EntryInfo, Error> {
        let e = self.get(id)?;
        Ok(info_of(*id, &e))
    }

    /// all entries, oldest first
    pub fn list(&self) -> Vec<EntryInfo> {
        let mut v: Vec<EntryInfo> = self.map.read().unwrap().iter().map(|(id, e)| info_of(*id, e)).collect();
        v.sort_by_key(|e| (e.created_ms, e.id));
        v
    }

    pub fn len(&self) -> usize {
        self.map.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn info_of(id: Uuid, e: &Entry) -> EntryInfo {
    EntryInfo {
        id,
        name: e.name.clone(),
        summary: e.value.summary().to_string(),
        metadata: e.metadata.clone(),
        nbytes: e.value.nbytes() as u64,
        created_ms: e.created_ms,
        updated_ms: e.updated_ms,
    }
}
