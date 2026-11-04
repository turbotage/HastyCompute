//! gRPC implementation of `hasty.v2.Hasty`. Value (de)compression runs on the blocking pool, connected to the
//! async streams by channels: decoding starts with the first chunk, encoded chunks are sent while encoding.

use std::pin::Pin;
use std::sync::Arc;

use hasty_value::{slice, Codec, Decoder, Options, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::{BroadcastStream, ReceiverStream};
use tokio_stream::{Stream, StreamExt};
use tonic::{Request, Response, Status, Streaming};
use uuid::Uuid;

use crate::bank::{Bank, EntryInfo};
use crate::commands::{Ctx, Registry};
use crate::pb::hasty_server::Hasty;
use crate::pb::*;
use crate::Error;

pub struct Service {
    pub bank: Arc<Bank>,
    pub registry: Arc<Registry>,
}

impl From<Error> for Status {
    fn from(e: Error) -> Self {
        match e {
            Error::NotFound(_) => Status::not_found(e.to_string()),
            Error::Invalid(_) | Error::Value(_) => Status::invalid_argument(e.to_string()),
            _ => Status::internal(e.to_string()),
        }
    }
}

pub fn uuid_of(id: &Option<Id>) -> Result<Uuid, Status> {
    let b = id.as_ref().map(|i| i.value.as_slice()).unwrap_or(&[]);
    Uuid::from_slice(b).map_err(|_| Status::invalid_argument("id must be 16 bytes"))
}

pub fn pid(u: &Uuid) -> Option<Id> {
    Some(Id { value: u.as_bytes().to_vec() })
}

pub fn entry_pb(e: EntryInfo) -> Entry {
    Entry { id: pid(&e.id), name: e.name, summary: e.summary, metadata: e.metadata, nbytes: e.nbytes, created_ms: e.created_ms, updated_ms: e.updated_ms }
}

pub fn options_of(enc: &Option<Encoding>) -> Result<Options, Status> {
    let e = enc.clone().unwrap_or_default();
    let codec = Codec::parse(&e.codec, e.level).map_err(|x| Status::invalid_argument(x.to_string()))?;
    let mut o = Options { codec, delta: e.delta, ..Options::default() };
    if e.block_size > 0 {
        o.block_size = e.block_size as usize;
    }
    Ok(o)
}

/// decode a value from chunks fed through `rx`, on the blocking pool
fn spawn_decoder() -> (mpsc::Sender<Vec<u8>>, tokio::task::JoinHandle<Result<Value, Error>>) {
    let (tx, mut rx) = mpsc::channel::<Vec<u8>>(16);
    let h = tokio::task::spawn_blocking(move || {
        let mut d = Decoder::new();
        while let Some(c) = rx.blocking_recv() {
            d.push(&c)?;
        }
        Ok(d.finish()?)
    });
    (tx, h)
}

type ChunkStream = Pin<Box<dyn Stream<Item = Result<Chunk, Status>> + Send>>;
type EventStream = Pin<Box<dyn Stream<Item = Result<Event, Status>> + Send>>;

impl Service {
    fn store(&self, h: &PutHeader, v: Value) -> Result<Uuid, Status> {
        match &h.id {
            Some(id) if !id.value.is_empty() => {
                let u = uuid_of(&h.id)?;
                self.bank.write(&u, &h.slice, v)?;
                Ok(u)
            }
            _ => {
                if !h.metadata.is_empty() {
                    serde_json::from_str::<serde_json::Value>(&h.metadata).map_err(|e| Status::invalid_argument(format!("metadata is not JSON: {e}")))?;
                }
                Ok(self.bank.put(&h.name, v, &h.metadata))
            }
        }
    }
}

#[tonic::async_trait]
impl Hasty for Service {
    async fn put(&self, req: Request<Streaming<PutRequest>>) -> Result<Response<PutResponse>, Status> {
        let mut s = req.into_inner();
        let header = match s.next().await {
            Some(Ok(PutRequest { part: Some(put_request::Part::Header(h)) })) => h,
            Some(Err(e)) => return Err(e),
            _ => return Err(Status::invalid_argument("the first message must be the header")),
        };
        let (tx, dec) = spawn_decoder();
        while let Some(m) = s.next().await {
            match m?.part {
                Some(put_request::Part::Data(d)) => {
                    if tx.send(d).await.is_err() {
                        break; // decoder failed; its error is reported below
                    }
                }
                _ => return Err(Status::invalid_argument("only data after the header")),
            }
        }
        drop(tx);
        let v = dec.await.map_err(|e| Status::internal(e.to_string()))??;
        Ok(Response::new(PutResponse { id: pid(&self.store(&header, v)?) }))
    }

    async fn put_unary(&self, req: Request<PutUnaryRequest>) -> Result<Response<PutResponse>, Status> {
        let r = req.into_inner();
        let h = r.header.unwrap_or_default();
        let data = r.data;
        let v = tokio::task::spawn_blocking(move || hasty_value::decode(&data)).await.map_err(|e| Status::internal(e.to_string()))?.map_err(Error::from)?;
        Ok(Response::new(PutResponse { id: pid(&self.store(&h, v)?) }))
    }

    type GetStream = ChunkStream;
    async fn get(&self, req: Request<GetRequest>) -> Result<Response<ChunkStream>, Status> {
        let r = req.into_inner();
        let u = uuid_of(&r.id)?;
        let v = self.bank.value(&u)?;
        let o = options_of(&r.encoding)?;
        let path = r.slice;
        let (tx, rx) = mpsc::channel::<Result<Chunk, Status>>(8);
        tokio::task::spawn_blocking(move || {
            let res = (|| -> Result<(), Error> {
                let sel;
                let val: &Value = if path.is_empty() {
                    &v
                } else {
                    sel = slice::get(&v, &path)?;
                    &sel
                };
                hasty_value::encode(val, &o, &mut |c| tx.blocking_send(Ok(Chunk { data: c })).map_err(|_| hasty_value::Error::Format("client went away".into())))?;
                Ok(())
            })();
            if let Err(e) = res {
                let _ = tx.blocking_send(Err(e.into()));
            }
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }

    async fn list(&self, _: Request<Empty>) -> Result<Response<ListResponse>, Status> {
        Ok(Response::new(ListResponse { entries: self.bank.list().into_iter().map(entry_pb).collect() }))
    }

    async fn info(&self, req: Request<Id>) -> Result<Response<Entry>, Status> {
        let u = uuid_of(&Some(req.into_inner()))?;
        Ok(Response::new(entry_pb(self.bank.info(&u)?)))
    }

    async fn delete(&self, req: Request<Id>) -> Result<Response<Empty>, Status> {
        self.bank.delete(&uuid_of(&Some(req.into_inner()))?)?;
        Ok(Response::new(Empty {}))
    }

    async fn set_metadata(&self, req: Request<MetadataRequest>) -> Result<Response<Entry>, Status> {
        let r = req.into_inner();
        Ok(Response::new(entry_pb(self.bank.set_metadata(&uuid_of(&r.id)?, &r.metadata)?)))
    }

    async fn commands(&self, _: Request<Empty>) -> Result<Response<CommandsResponse>, Status> {
        Ok(Response::new(CommandsResponse { commands: self.registry.list().into_iter().map(|(name, doc)| CommandInfo { name, doc }).collect() }))
    }

    async fn execute(&self, req: Request<ExecuteRequest>) -> Result<Response<ExecuteResponse>, Status> {
        let r = req.into_inner();
        let inputs = r.inputs.iter().map(|i| uuid_of(&Some(i.clone())).and_then(|u| self.bank.value(&u).map_err(Status::from))).collect::<Result<Vec<_>, _>>()?;
        let (bank, reg) = (self.bank.clone(), self.registry.clone());
        let (cmd, opts) = (r.command.clone(), r.options.clone());
        let (msg, outs) = tokio::task::spawn_blocking(move || reg.run(&cmd, &Ctx { bank }, &opts, &inputs)).await.map_err(|e| Status::internal(e.to_string()))??;
        let name = if r.name.is_empty() { r.command.clone() } else { r.name.clone() };
        let n = outs.len();
        let outputs = outs.into_iter().enumerate().map(|(k, v)| Id { value: self.bank.put(&if n > 1 { format!("{name}[{k}]") } else { name.clone() }, v, "{}").as_bytes().to_vec() }).collect();
        Ok(Response::new(ExecuteResponse { outputs, message: msg }))
    }

    async fn show(&self, req: Request<ShowRequest>) -> Result<Response<Empty>, Status> {
        let r = req.into_inner();
        let u = uuid_of(&r.id)?;
        let e = self.bank.info(&u)?;
        self.bank.notify("show", Some(u), &e.name, &r.view);
        Ok(Response::new(Empty {}))
    }

    type WatchStream = EventStream;
    async fn watch(&self, _: Request<Empty>) -> Result<Response<EventStream>, Status> {
        let s = BroadcastStream::new(self.bank.subscribe()).filter_map(|e| e.ok()).map(|e| Ok(Event { kind: e.kind, id: e.id.as_ref().and_then(pid), name: e.name, message: e.message }));
        Ok(Response::new(Box::pin(s)))
    }
}
