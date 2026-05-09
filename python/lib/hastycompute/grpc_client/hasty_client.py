"""
gRPC client for HastyServer.
"""

import grpc
from hastycompute.grpc_client.gen import hasty_service_pb2 as pb
from hastycompute.grpc_client.gen import hasty_service_pb2_grpc as pb_grpc
from hastycompute.generic_value import GenericValue

_CHUNK_SIZE = 2 * 1024 * 1024

_CHANNEL_OPTIONS = [
    ('grpc.max_send_message_length', -1),
    ('grpc.max_receive_message_length', -1),
    ('grpc.http2.lookahead_bytes', 64 * 1024 * 1024),
    ('grpc.http2.initial_window_size', 64 * 1024 * 1024),
]


class HastyClient:
    def __init__(self, address: str = "localhost:50051"):
        self._channel = grpc.insecure_channel(address, options=_CHANNEL_OPTIONS)
        self._stub = pb_grpc.HastyServiceStub(self._channel)

    def close(self):
        self._channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # -----------------------------------------------------------------------
    # Push: serialize GenericValue, stream DataChunks → PushResponse.id (UUID)
    # -----------------------------------------------------------------------

    def push_value(self, gv: GenericValue) -> bytes:
        response = self._stub.Push(
            pb.DataChunk(data=chunk) for chunk in gv.iter_chunks(_CHUNK_SIZE)
        )
        if not response.success:
            raise RuntimeError(f"Push failed: {response.msg}")
        return response.id.value  # raw 16-byte UUID

    # -----------------------------------------------------------------------
    # Read: ReadRequest → stream ReadResponse (oneof header | data)
    # -----------------------------------------------------------------------

    def fetch_value(self, uuid_bytes: bytes, slice_info: str = "") -> GenericValue:
        request = pb.ReadRequest(
            id=pb.Uuid(value=uuid_bytes),
            slice_info=slice_info,
        )
        responses = self._stub.Read(request)

        # First message must be the header.
        first = next(iter(responses))
        if first.HasField('header') and not first.header.success:
            raise RuntimeError(f"Read failed: {first.header.msg}")

        def _chunks():
            # First response may carry data directly if header was in it.
            if first.HasField('data'):
                yield first.data.data
            for resp in responses:
                if resp.HasField('data'):
                    yield resp.data.data

        return GenericValue.deserialize_chunks(_chunks())

    # -----------------------------------------------------------------------
    # Write: stream WriteRequest (header then data chunks) → WriteResponse
    # -----------------------------------------------------------------------

    def write_value(self, uuid_bytes: bytes, gv: GenericValue,
                    slice_info: str = "") -> None:
        data = gv.serialize()

        def _messages():
            yield pb.WriteRequest(
                header=pb.WriteRequestHeader(
                    id=pb.Uuid(value=uuid_bytes),
                    slice_info=slice_info,
                )
            )
            for i in range(0, len(data), _CHUNK_SIZE):
                yield pb.WriteRequest(
                    data=pb.DataChunk(data=data[i:i + _CHUNK_SIZE])
                )

        response = self._stub.Write(_messages())
        if not response.success:
            raise RuntimeError(f"Write failed: {response.error_msg}")

    # -----------------------------------------------------------------------
    # Delete: Uuid → DeleteResponse
    # -----------------------------------------------------------------------

    def delete_value(self, uuid_bytes: bytes) -> None:
        response = self._stub.Delete(pb.Uuid(value=uuid_bytes))
        if not response.success:
            raise RuntimeError(f"Delete failed: {response.msg}")

    # -----------------------------------------------------------------------
    # ExecuteCommand: run a registered command on bank values
    # -----------------------------------------------------------------------

    def execute(self, function_id: int, input_uuids: list[bytes],
                options: str = "") -> list[bytes]:
        request = pb.ExecuteCommandRequest(
            function_id=function_id,
            options=options,
            input_ids=[pb.Uuid(value=u) for u in input_uuids],
        )
        response = self._stub.ExecuteCommand(request)
        if not response.success:
            raise RuntimeError(f"ExecuteCommand failed: {response.error_msg}")
        return [u.value for u in response.output_ids]

    # -----------------------------------------------------------------------
    # BankQuery: query things about the bank
    # -----------------------------------------------------------------------

    def list_uuids(self) -> list[bytes]:
        request = pb.BankQueryRequest(query_type=0)  # LIST_UUIDS = 0
        response = self._stub.BankQuery(request)
        if not response.success:
            raise RuntimeError(f"BankQuery failed: {response.error_msg}")
        return [u.value for u in response.ids_in_bank]

    # -----------------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------------

    def read_metadata(self, uuid_bytes: bytes) -> str | None:
        response = self._stub.ReadMetadata(pb.MetadataReadRequest(id=pb.Uuid(value=uuid_bytes)))
        if not response.success:
            return None
        return response.metadata_string

    def write_metadata(self, uuid_bytes: bytes, metadata: str) -> None:
        response = self._stub.WriteMetadata(pb.MetadataWriteRequest(
            id=pb.Uuid(value=uuid_bytes),
            metadata_string=metadata,
        ))
        if not response.success:
            raise RuntimeError(f"WriteMetadata failed: {response.msg}")

    def delete_metadata(self, uuid_bytes: bytes) -> None:
        response = self._stub.DeleteMetadata(pb.MetadataDeleteRequest(id=pb.Uuid(value=uuid_bytes)))
        if not response.success:
            raise RuntimeError(f"DeleteMetadata failed: {response.msg}")
