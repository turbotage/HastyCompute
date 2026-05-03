module;

#define CPPHTTPLIB_OPENSSL_SUPPORT
#undef CPPHTTPLIB_ZLIB_SUPPORT
#include <httplib.h>

#include <grpcpp/grpcpp.h>
#include <grpcpp/generic/generic_stub.h>

export module hasty_server_mod:http_server;

import std;
import hasty_util_mod;
import :grpc_server;

namespace hasty {
namespace server {

// ── gRPC-web framing ─────────────────────────────────────────────────────────
//
// Every gRPC message is prefixed with a 5-byte frame header:
//   [1 byte flags | 4 bytes big-endian length]
// flags: 0x00 = data frame, 0x80 = trailer frame (text key:value\r\n pairs)
//
// gRPC-web encodes ALL stream types this way:
//   Unary request/response    — single data frame each
//   Client-streaming request  — N consecutive data frames in the HTTP body
//   Server-streaming response — N consecutive data frames in the HTTP body
//
// The proxy is therefore stream-type-agnostic: it parses all request frames,
// sends each as a gRPC message, collects all response messages, and serialises
// them back as frames.

namespace grpc_web {

static uint32_t read_be32(const char* p) {
    return (static_cast<uint32_t>(static_cast<uint8_t>(p[0])) << 24)
         | (static_cast<uint32_t>(static_cast<uint8_t>(p[1])) << 16)
         | (static_cast<uint32_t>(static_cast<uint8_t>(p[2])) <<  8)
         |  static_cast<uint32_t>(static_cast<uint8_t>(p[3]));
}

static std::string make_frame(uint8_t flags, std::string_view payload) {
    std::string f(5 + payload.size(), '\0');
    f[0] = static_cast<char>(flags);
    uint32_t n = static_cast<uint32_t>(payload.size());
    f[1] = static_cast<char>((n >> 24) & 0xFF);
    f[2] = static_cast<char>((n >> 16) & 0xFF);
    f[3] = static_cast<char>((n >>  8) & 0xFF);
    f[4] = static_cast<char>( n        & 0xFF);
    payload.copy(f.data() + 5, payload.size());
    return f;
}

// Parse all data frames from body into a list of ByteBuffers (one per message).
// Trailer frames (flags & 0x80) are skipped.
static std::vector<grpc::ByteBuffer> parse_frames(std::string_view body) {
    std::vector<grpc::ByteBuffer> out;
    while (body.size() >= 5) {
        uint8_t flags = static_cast<uint8_t>(body[0]);
        if (flags & 0x80) break;                         // trailer; done
        uint32_t len = read_be32(body.data() + 1);
        if (body.size() < 5u + len) break;              // truncated; bail
        auto payload = body.substr(5, len);
        grpc::Slice s(payload.data(), payload.size());
        out.emplace_back(&s, 1);
        body = body.substr(5 + len);
    }
    return out;
}

// Serialise a gRPC ByteBuffer into a raw string.
static std::string dump_buffer(grpc::ByteBuffer& buf) {
    std::vector<grpc::Slice> slices;
    buf.Dump(&slices);
    std::string out;
    for (auto& s : slices)
        out.append(reinterpret_cast<const char*>(s.begin()), s.size());
    return out;
}

} // namespace grpc_web


// ── HttpGrpcProxy ─────────────────────────────────────────────────────────────
//
// Translates one HTTP gRPC-web request into gRPC bidi-streaming call, then
// collects all response messages and returns the gRPC-web response body.
//
// Using bidi streaming (GenericStub::PrepareCall) as the transport makes
// the proxy handle all three RPC flavours uniformly:
//
//   Unary            PushValue / FetchValue request   — 1 write, 1 read
//   Client-streaming PushValue / WriteValue request   — N writes, 1 read
//   Server-streaming FetchValue response              — 1 write, N reads
//
// The server determines how many messages it reads/writes; the proxy just
// writes all request frames and reads until the server closes.

export class HttpGrpcProxy {
    std::shared_ptr<grpc::Channel> _ch;

public:
    explicit HttpGrpcProxy(std::shared_ptr<grpc::Channel> ch)
        : _ch(std::move(ch)) {}

    // service_method: full gRPC path, e.g. "/hasty.HastyService/FetchValue"
    // body:           raw HTTP body (one or more gRPC-web frames)
    // Returns gRPC-web response body on success, empty on framing error.
    std::string call(const std::string& service_method, const std::string& body) {
        auto req_bufs = grpc_web::parse_frames(body);
        if (req_bufs.empty()) return {};

        grpc::GenericStub stub(_ch);
        grpc::ClientContext ctx;
        ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(60));

        grpc::CompletionQueue cq;

        // Use bidi stream regardless of actual RPC type — works for all variants.
        auto stream = stub.PrepareCall(&ctx, service_method, &cq);

        void* tag = nullptr;
        bool ok   = false;

        // Start
        stream->StartCall((void*)1);
        cq.Next(&tag, &ok);
        if (!ok) return grpc_web_error(grpc::StatusCode::UNAVAILABLE, "StartCall failed");

        // Write all request frames
        for (auto& buf : req_bufs) {
            stream->Write(buf, (void*)2);
            cq.Next(&tag, &ok);
            if (!ok) return grpc_web_error(grpc::StatusCode::UNAVAILABLE, "Write failed");
        }
        stream->WritesDone((void*)3);
        cq.Next(&tag, &ok);  // ok may be false if server closed early; continue

        // Read all response frames until server closes
        std::string result;
        while (true) {
            grpc::ByteBuffer resp;
            stream->Read(&resp, (void*)4);
            cq.Next(&tag, &ok);
            if (!ok) break;  // server sent all messages
            result += grpc_web::make_frame(0x00, grpc_web::dump_buffer(resp));
        }

        grpc::Status status;
        stream->Finish(&status, (void*)5);
        cq.Next(&tag, &ok);
        cq.Shutdown();

        if (!status.ok())
            return grpc_web_error(status.error_code(), status.error_message());

        result += grpc_web::make_frame(0x80, "grpc-status:0\r\n");
        return result;
    }

private:
    static std::string grpc_web_error(grpc::StatusCode code, const std::string& msg) {
        std::string trailers = "grpc-status:" + std::to_string(static_cast<int>(code))
                             + "\r\ngrpc-message:" + msg + "\r\n";
        return grpc_web::make_frame(0x80, trailers);
    }
};


// ── HttpServer ────────────────────────────────────────────────────────────────
//
// Serves static files from plotting_website/ at "/" and routes gRPC-web
// POST requests to the running gRPC server via a loopback channel.
//
// Route priority (httplib processes in registration order):
//   1. CORS OPTIONS preflight — always first
//   2. gRPC-web POST /<Pkg>.<Svc>/<Method>
//   3. Static file mount — last, catches everything else

export class HttpServer {
    httplib::SSLServer _srv;
    std::string        _static_root;
    HttpGrpcProxy   _proxy;
    int             _port;
    std::thread     _thread;
    SPtr<GrpcServerHandle> _grpc_handle;

public:
    // port        — HTTP listen port (e.g. 8080)
    // static_root — path to plotting_website/ directory
    // grpc_handle — running gRPC server; HttpServer creates a loopback channel to it
    HttpServer(int port, std::string static_root, SPtr<GrpcServerHandle> grpc_handle,
               std::string cert_path, std::string key_path)
        : _srv(cert_path.c_str(), key_path.c_str())
        , _static_root(std::move(static_root))
        , _proxy(grpc::CreateChannel(
              loopback_address(grpc_handle->address()),
              grpc::InsecureChannelCredentials()))
        , _port(port)
        , _grpc_handle(std::move(grpc_handle))
    {
        setup_routes();
    }

    void start() {
        if (!_srv.is_valid())
            throw std::runtime_error("HttpServer: SSLServer is not valid — check cert/key paths");
        _thread = std::thread([this] { _srv.listen("0.0.0.0", _port); });
    }

    void stop() {
        _srv.stop();
        if (_thread.joinable()) _thread.join();
    }

    bool is_running() const { return _srv.is_running(); }

private:
    // Replace "0.0.0.0" in the gRPC listen address with "localhost" so the
    // outgoing channel reaches the same server.
    static std::string loopback_address(const std::string& addr) {
        std::string a = addr;
        auto pos = a.find("0.0.0.0");
        if (pos != std::string::npos) a.replace(pos, 7, "localhost");
        return a;
    }

    void setup_routes() {
        // CORS preflight — registered before everything else
        _srv.Options(".*", [](const httplib::Request&, httplib::Response& res) {
            res.set_header("Access-Control-Allow-Origin",  "*");
            res.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
            res.set_header("Access-Control-Allow-Headers",
                "Content-Type, x-grpc-web, x-user-agent, grpc-timeout");
            res.set_header("Access-Control-Max-Age", "86400");
            res.status = 204;
        });

        // gRPC-web proxy: POST /<Package>.<Service>/<Method>
        _srv.Post(
            R"(/[A-Za-z0-9_.]+/[A-Za-z0-9_]+)",
            [this](const httplib::Request& req, httplib::Response& res) {
                auto ct = req.get_header_value("Content-Type");
                bool grpc_web = ct.find("application/grpc-web") != std::string::npos
                             || ct.find("application/grpc+proto") != std::string::npos;
                if (!grpc_web) { res.status = 415; return; }

                std::string resp_body = _proxy.call(req.path, req.body);
                if (resp_body.empty()) { res.status = 400; return; }

                res.set_content(resp_body, "application/grpc-web+proto");
                res.set_header("Access-Control-Allow-Origin",   "*");
                res.set_header("Access-Control-Expose-Headers", "grpc-status,grpc-message");
                res.set_header("x-grpc-web", "1");
                res.set_header("Trailer",     "grpc-status,grpc-message");
            }
        );

        // Static files — lowest priority
        if (!_static_root.empty() && !_srv.set_mount_point("/", _static_root)) {
            _srv.Get(".*", [this](const httplib::Request&, httplib::Response& res) {
                res.status = 404;
                res.set_content("plotting_website not found at: " + _static_root, "text/plain");
            });
        }
    }
};

}
}