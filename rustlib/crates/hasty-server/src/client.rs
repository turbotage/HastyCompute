//! Async gRPC client of a hasty server (TLS verified against the local CA).

use hasty_value::{Decoder, Options, Value};
use tokio_stream::StreamExt;
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint};
use uuid::Uuid;

use crate::pb::hasty_client::HastyClient;
use crate::pb::*;
use crate::service::{pid, uuid_of};
use crate::Error;

pub struct Client {
    inner: HastyClient<Channel>,
}

fn st(s: tonic::Status) -> Error {
    Error::Invalid(format!("{}: {}", s.code(), s.message()))
}

impl Client {
    /// `url` like `https://host:8443`; `ca_pem` = the server's CA (data/hasty/certs/ca.pem) for https
    pub async fn connect(url: &str, ca_pem: Option<&str>) -> Result<Self, Error> {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let mut ep = Endpoint::from_shared(url.to_string()).map_err(|e| Error::Invalid(e.to_string()))?.connect_timeout(std::time::Duration::from_millis(1500));
        if url.starts_with("https") {
            let mut tls = ClientTlsConfig::new();
            if let Some(ca) = ca_pem {
                tls = tls.ca_certificate(Certificate::from_pem(ca));
            }
            ep = ep.tls_config(tls).map_err(|e| Error::Tls(e.to_string()))?;
        }
        let ch = ep.connect().await.map_err(|e| Error::Invalid(format!("connect {url}: {e}")))?;
        Ok(Self { inner: HastyClient::new(ch).max_decoding_message_size(1 << 30).max_encoding_message_size(1 << 30) })
    }

    /// upload `value` (streamed, compressed with `o`) as a new entry
    pub async fn put(&mut self, name: &str, value: &Value, metadata: &str, o: &Options) -> Result<Uuid, Error> {
        let mut chunks = Vec::new();
        hasty_value::encode(value, o, &mut |c| {
            chunks.push(c);
            Ok(())
        })?;
        let header = PutRequest { part: Some(put_request::Part::Header(PutHeader { id: None, slice: String::new(), name: name.into(), metadata: metadata.into() })) };
        let msgs = std::iter::once(header).chain(chunks.into_iter().map(|d| PutRequest { part: Some(put_request::Part::Data(d)) }));
        let r = self.inner.put(tokio_stream::iter(msgs)).await.map_err(st)?.into_inner();
        uuid_of(&r.id).map_err(st)
    }

    pub async fn get(&mut self, id: &Uuid, slice: &str, codec: &str) -> Result<Value, Error> {
        let req = GetRequest { id: pid(id), slice: slice.into(), encoding: Some(Encoding { codec: codec.into(), ..Default::default() }) };
        let mut s = self.inner.get(req).await.map_err(st)?.into_inner();
        let mut d = Decoder::new();
        while let Some(c) = s.next().await {
            d.push(&c.map_err(st)?.data)?;
        }
        Ok(d.finish()?)
    }

    pub async fn list(&mut self) -> Result<Vec<Entry>, Error> {
        Ok(self.inner.list(Empty {}).await.map_err(st)?.into_inner().entries)
    }

    pub async fn delete(&mut self, id: &Uuid) -> Result<(), Error> {
        self.inner.delete(Id { value: id.as_bytes().to_vec() }).await.map_err(st)?;
        Ok(())
    }

    pub async fn execute(&mut self, command: &str, options: &str, inputs: &[Uuid]) -> Result<(Vec<Uuid>, String), Error> {
        let r = self.inner.execute(ExecuteRequest { command: command.into(), options: options.into(), inputs: inputs.iter().map(|u| Id { value: u.as_bytes().to_vec() }).collect(), name: String::new() }).await.map_err(st)?.into_inner();
        Ok((r.outputs.iter().map(|i| uuid_of(&Some(i.clone()))).collect::<Result<Vec<_>, _>>().map_err(st)?, r.message))
    }

    pub async fn show(&mut self, id: &Uuid, view: &str) -> Result<(), Error> {
        self.inner.show(ShowRequest { id: pid(id), view: view.into() }).await.map_err(st)?;
        Ok(())
    }
}
