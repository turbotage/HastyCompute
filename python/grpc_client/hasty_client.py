"""
gRPC client for HastyServer.
"""

import grpc
from gen import hasty_service_pb2 as pb
from gen import hasty_service_pb2_grpc as pb_grpc
from generic_value import GenericValue

_CHUNK_SIZE = 2 * 1024 * 1024

#options=[
#            ('grpc.max_send_message_length', -1),
#            ('grpc.max_receive_message_length', -1),
#            ('grpc.http2.lookahead_bytes', 64 * 1024 * 1024),              # 64 MiB outbound write buffer
#            ('grpc.http2.initial_window_size', 64 * 1024 * 1024),           # 64 MiB per-stream receive window
#            ('grpc.http2.initial_connection_window_size', 64 * 1024 * 1024), # 64 MiB connection-level window
#            ('grpc.http2.bdp_probe', 0),                                     # disable BDP auto-tuning
#        ],

class HastyClient:
    def __init__(self, address: str = "localhost:50051"):
        self._channel = grpc.insecure_channel(
            address,
            options=[
                ('grpc.max_send_message_length', -1),
                ('grpc.max_receive_message_length', -1),
                ('grpc.http2.lookahead_bytes', 64 * 1024 * 1024),              # 64 MiB outbound write buffer
                ('grpc.http2.initial_window_size', 64 * 1024 * 1024)           # 64 MiB per-stream receive window
            ],
        )
        self._stub = pb_grpc.HastyServiceStub(self._channel)

    def close(self):
        self._channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # -----------------------------------------------------------------------
    # PushValue: serialize GenericValue, stream to server, get back UUID bytes
    # -----------------------------------------------------------------------

    def push_value(self, gv: GenericValue) -> bytes:
        response = self._stub.PushValue(
            pb.DataChunk(data=chunk) for chunk in gv.iter_chunks(_CHUNK_SIZE)
        )
        return response.value  # raw 16-byte UUID

    # -----------------------------------------------------------------------
    # FetchValue: request by UUID, receive chunks, deserialize
    # -----------------------------------------------------------------------

    def fetch_value(self, uuid_bytes: bytes) -> GenericValue:
        request = pb.FetchRequest(id=pb.Uuid(value=uuid_bytes))
        chunks = self._stub.FetchValue(request)
        return GenericValue.deserialize_chunks(chunk.data for chunk in chunks)

    # -----------------------------------------------------------------------
    # WriteValue: write to existing UUID (full overwrite for now)
    # -----------------------------------------------------------------------

    def write_value(self, uuid_bytes: bytes, gv: GenericValue,
                    slice_info: str = "") -> None:
        data = gv.serialize()

        def _messages():
            yield pb.WriteMessage(
                header=pb.WriteHeader(
                    id=pb.Uuid(value=uuid_bytes),
                    slice_info=slice_info,
                )
            )
            for i in range(0, len(data), _CHUNK_SIZE):
                yield pb.WriteMessage(
                    data=pb.DataChunk(data=data[i:i + _CHUNK_SIZE])
                )

        response = self._stub.WriteValue(_messages())
        if not response.success:
            raise RuntimeError(f"WriteValue failed: {response.error_msg}")

    # -----------------------------------------------------------------------
    # DeleteValue: free a bank entry by UUID
    # -----------------------------------------------------------------------

    def delete_value(self, uuid_bytes: bytes) -> None:
        response = self._stub.DeleteValue(pb.Uuid(value=uuid_bytes))
        if not response.success:
            raise RuntimeError(f"DeleteValue failed: {response.error_msg}")

    # -----------------------------------------------------------------------
    # Execute: run a registered command on bank values
    # -----------------------------------------------------------------------

    def execute(self, function_id: int, input_uuids: list[bytes],
                options: str = "") -> list[bytes]:
        cmd = pb.ExecuteCommand(
            function_id=function_id,
            options=options,
            input_ids=[pb.Uuid(value=u) for u in input_uuids],
        )
        response = self._stub.Execute(cmd)
        if not response.success:
            raise RuntimeError(f"Execute failed: {response.error_msg}")
        return [u.value for u in response.output_ids]
