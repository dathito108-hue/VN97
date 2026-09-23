package ai.vn97.platform

import ai.vn97.runtime.VN97CapabilityPackageParser
import ai.vn97.runtime.VN97ParsedCapabilityPackage
import java.io.BufferedInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.net.IDN
import java.net.Inet4Address
import java.net.Inet6Address
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.Socket
import java.net.URI
import java.nio.file.FileAlreadyExistsException
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.StandardOpenOption
import javax.net.ssl.SNIHostName
import javax.net.ssl.SSLSocket
import javax.net.ssl.SSLSocketFactory

data class VN97FetchedCapabilityArtifact(
    val canonicalUrl: String,
    val packageSha256: String,
    val sizeBytes: Long,
    val capabilityId: String,
    val capabilityVersion: Long,
    val kind: String,
) {
    init {
        require(canonicalUrl.startsWith("https://"))
        require(
            packageSha256.length == 64 &&
                packageSha256.all {
                    it in "0123456789abcdef"
                }
        )
        require(sizeBytes > 0L)
        require(capabilityId.isNotBlank())
        require(capabilityVersion > 0L)
        require(kind.isNotBlank())
    }

    fun wireResult(): String =
        buildString {
            append("VN97ARTFETCH1")
            append('|')
            append(packageSha256)
            append('|')
            append(sizeBytes)
            append('|')
            append(capabilityId)
            append('|')
            append(capabilityVersion)
            append('|')
            append(kind)
        }

    companion object {
        fun parseWireResult(
            value: String,
        ): VN97FetchedCapabilityArtifact {
            val fields = value.split('|')
            require(
                fields.size == 6 &&
                    fields[0] == "VN97ARTFETCH1"
            ) {
                "remote capability artifact result is malformed"
            }
            val sha = fields[1]
            require(
                sha.length == 64 &&
                    sha.all {
                        it in "0123456789abcdef"
                    }
            ) {
                "remote capability artifact digest is invalid"
            }
            val size = fields[2].toLongOrNull()
                ?: throw IllegalArgumentException(
                    "remote capability artifact size is invalid"
                )
            val version = fields[4].toLongOrNull()
                ?: throw IllegalArgumentException(
                    "remote capability artifact version is invalid"
                )
            return VN97FetchedCapabilityArtifact(
                canonicalUrl =
                    "https://receipt.invalid/",
                packageSha256 = sha,
                sizeBytes = size,
                capabilityId = fields[3],
                capabilityVersion = version,
                kind = fields[5],
            )
        }
    }
}

fun interface VN97CapabilityArtifactTransport {
    fun get(
        endpoint: URI,
        maxResponseBytes: Int,
        connectTimeoutMs: Int,
        readTimeoutMs: Int,
    ): ByteArray
}

data class VN97CapabilityArtifactFetchPolicy(
    val maxResponseBytes: Int =
        16 * 1024 * 1024,
    val connectTimeoutMs: Int = 8_000,
    val readTimeoutMs: Int = 15_000,
) {
    init {
        require(
            maxResponseBytes in
                1024..64 * 1024 * 1024
        ) {
            "remote capability response byte bound is invalid"
        }
        require(connectTimeoutMs in 250..30_000) {
            "remote capability connect timeout is invalid"
        }
        require(readTimeoutMs in 250..60_000) {
            "remote capability read timeout is invalid"
        }
    }
}

class VN97PinnedHttpsCapabilityTransport :
    VN97CapabilityArtifactTransport {
    override fun get(
        endpoint: URI,
        maxResponseBytes: Int,
        connectTimeoutMs: Int,
        readTimeoutMs: Int,
    ): ByteArray {
        val canonical =
            canonicalCapabilityHttpsUrl(
                endpoint.toASCIIString()
            )
        val uri = URI(canonical)
        val host = checkNotNull(uri.host)
        val addresses =
            InetAddress.getAllByName(host).toList()
        require(addresses.isNotEmpty()) {
            "remote capability host resolved to no addresses"
        }
        require(
            addresses.all(
                ::isGloballyRoutableAddress
            )
        ) {
            "remote capability host resolved to a non-global address"
        }
        val pinned = addresses
            .sortedBy {
                it.hostAddress ?: ""
            }
            .first()
        val port =
            if (uri.port == -1) 443 else uri.port
        require(port == 443) {
            "remote capability HTTPS port must be 443"
        }

        Socket().use { raw ->
            raw.connect(
                InetSocketAddress(pinned, port),
                connectTimeoutMs,
            )
            raw.soTimeout = readTimeoutMs
            val ssl = (
                SSLSocketFactory.getDefault()
                    as SSLSocketFactory
            ).createSocket(
                raw,
                host,
                port,
                true,
            ) as SSLSocket
            ssl.use { socket ->
                configureTls(socket, host)
                socket.startHandshake()
                writeRequest(socket, uri, host)
                return readResponse(
                    socket,
                    maxResponseBytes,
                )
            }
        }
    }

    private fun configureTls(
        socket: SSLSocket,
        host: String,
    ) {
        val parameters = socket.sslParameters
        parameters.endpointIdentificationAlgorithm =
            "HTTPS"
        if (!isIpLiteral(host)) {
            parameters.serverNames =
                listOf(SNIHostName(host))
        }
        socket.sslParameters = parameters
    }

    private fun writeRequest(
        socket: SSLSocket,
        uri: URI,
        host: String,
    ) {
        val rawPath =
            uri.rawPath
                ?.takeIf { it.isNotEmpty() }
                ?: "/"
        val target =
            if (uri.rawQuery.isNullOrEmpty()) {
                rawPath
            } else {
                "$rawPath?${uri.rawQuery}"
            }
        val request = buildString {
            append("GET ")
            append(target)
            append(" HTTP/1.1\r\n")
            append("Host: ")
            append(host)
            append("\r\n")
            append(
                "Accept: application/octet-stream, application/vnd.vn97.capability\r\n"
            )
            append(
                "Accept-Encoding: identity\r\n"
            )
            append("Cache-Control: no-cache\r\n")
            append("Connection: close\r\n")
            append("User-Agent: VN97/1\r\n")
            append("\r\n")
        }.toByteArray(Charsets.US_ASCII)
        val output = socket.outputStream
        output.write(request)
        output.flush()
    }

    private fun readResponse(
        socket: SSLSocket,
        maxResponseBytes: Int,
    ): ByteArray {
        val input =
            BufferedInputStream(socket.inputStream)
        val status =
            readAsciiLine(
                input,
                MAX_HEADER_LINE_BYTES,
            )
        require(
            status == "HTTP/1.1 200 OK" ||
                status == "HTTP/1.0 200 OK"
        ) {
            "remote capability HTTPS status is not 200"
        }

        val headers =
            linkedMapOf<String, String>()
        var count = 0
        while (true) {
            val line =
                readAsciiLine(
                    input,
                    MAX_HEADER_LINE_BYTES,
                )
            if (line.isEmpty()) break
            count += 1
            require(count <= MAX_HEADER_COUNT) {
                "remote capability response has too many headers"
            }
            val split = line.indexOf(':')
            require(split > 0) {
                "remote capability response header is malformed"
            }
            val name =
                line.substring(0, split)
                    .trim()
                    .lowercase()
            val value =
                line.substring(split + 1).trim()
            require(
                name.isNotEmpty() &&
                    name.all {
                        it in 'a'..'z' ||
                            it in '0'..'9' ||
                            it == '-'
                    }
            ) {
                "remote capability response header name is invalid"
            }
            require(
                !headers.containsKey(name)
            ) {
                "remote capability response contains duplicate header"
            }
            headers[name] = value
        }

        val encoding =
            headers["content-encoding"]
        require(
            encoding == null ||
                encoding.equals(
                    "identity",
                    ignoreCase = true,
                )
        ) {
            "remote capability content encoding is unsupported"
        }

        val transfer =
            headers["transfer-encoding"]
        val contentLength =
            headers["content-length"]?.let {
                require(
                    it.isNotEmpty() &&
                        it.all { ch ->
                            ch in '0'..'9'
                        }
                ) {
                    "remote capability content length is invalid"
                }
                it.toLongOrNull()
                    ?: throw IllegalArgumentException(
                        "remote capability content length is outside range"
                    )
            }
        require(
            contentLength == null ||
                contentLength in
                    1L..maxResponseBytes.toLong()
        ) {
            "remote capability response exceeds byte bound"
        }

        val body = when {
            transfer == null ->
                readBoundedBody(
                    input,
                    contentLength,
                    maxResponseBytes,
                )
            transfer.equals(
                "chunked",
                ignoreCase = true,
            ) -> {
                require(contentLength == null) {
                    "remote capability response mixes chunked and content-length"
                }
                readChunkedBody(
                    input,
                    maxResponseBytes,
                )
            }
            else ->
                throw IllegalArgumentException(
                    "remote capability transfer encoding is unsupported"
                )
        }
        require(body.isNotEmpty()) {
            "remote capability response is empty"
        }
        return body
    }

    private fun readBoundedBody(
        input: BufferedInputStream,
        contentLength: Long?,
        maxResponseBytes: Int,
    ): ByteArray {
        val output =
            ByteArrayOutputStream(
                contentLength
                    ?.coerceAtMost(
                        maxResponseBytes.toLong()
                    )
                    ?.toInt()
                    ?: minOf(
                        maxResponseBytes,
                        16 * 1024,
                    )
            )
        val buffer = ByteArray(16 * 1024)
        var total = 0
        while (true) {
            val read = input.read(buffer)
            if (read < 0) break
            require(read > 0) {
                "remote capability response read made no progress"
            }
            total = Math.addExact(total, read)
            require(total <= maxResponseBytes) {
                "remote capability response exceeds byte bound"
            }
            output.write(buffer, 0, read)
        }
        if (contentLength != null) {
            require(
                total.toLong() == contentLength
            ) {
                "remote capability response length changed"
            }
        }
        return output.toByteArray()
    }

    private fun readChunkedBody(
        input: BufferedInputStream,
        maxResponseBytes: Int,
    ): ByteArray {
        val output =
            ByteArrayOutputStream(
                minOf(
                    maxResponseBytes,
                    16 * 1024,
                )
            )
        var total = 0
        while (true) {
            val line =
                readAsciiLine(
                    input,
                    MAX_HEADER_LINE_BYTES,
                )
            require(';' !in line) {
                "remote capability chunk extensions are unsupported"
            }
            require(
                line.isNotEmpty() &&
                    line.all {
                        it in '0'..'9' ||
                            it in 'a'..'f' ||
                            it in 'A'..'F'
                    }
            ) {
                "remote capability chunk size is invalid"
            }
            val size =
                line.toLongOrNull(16)
                    ?: throw IllegalArgumentException(
                        "remote capability chunk size is outside range"
                    )
            require(
                size in
                    0L..maxResponseBytes.toLong()
            ) {
                "remote capability chunk exceeds byte bound"
            }
            if (size == 0L) {
                require(
                    readAsciiLine(
                        input,
                        MAX_HEADER_LINE_BYTES,
                    ).isEmpty()
                ) {
                    "remote capability response trailers are unsupported"
                }
                break
            }
            total =
                Math.addExact(
                    total,
                    size.toInt(),
                )
            require(total <= maxResponseBytes) {
                "remote capability response exceeds byte bound"
            }
            copyExactly(
                input,
                output,
                size.toInt(),
            )
            require(
                input.read() == '\r'.code
            ) {
                "remote capability chunk terminator is invalid"
            }
            require(
                input.read() == '\n'.code
            ) {
                "remote capability chunk terminator is invalid"
            }
        }
        return output.toByteArray()
    }

    private fun copyExactly(
        input: BufferedInputStream,
        output: ByteArrayOutputStream,
        count: Int,
    ) {
        var remaining = count
        val buffer =
            ByteArray(
                minOf(16 * 1024, count)
            )
        while (remaining > 0) {
            val read =
                input.read(
                    buffer,
                    0,
                    minOf(
                        buffer.size,
                        remaining,
                    ),
                )
            require(read > 0) {
                "remote capability chunk ended early"
            }
            output.write(buffer, 0, read)
            remaining -= read
        }
    }

    private fun readAsciiLine(
        input: BufferedInputStream,
        maxBytes: Int,
    ): String {
        val output =
            ByteArrayOutputStream()
        while (true) {
            val value = input.read()
            require(value >= 0) {
                "remote capability response ended inside header"
            }
            if (value == '\r'.code) {
                require(
                    input.read() == '\n'.code
                ) {
                    "remote capability response line ending is invalid"
                }
                return String(
                    output.toByteArray(),
                    Charsets.US_ASCII,
                )
            }
            require(value != '\n'.code) {
                "remote capability response contains bare LF"
            }
            require(value in 0x20..0x7e) {
                "remote capability response header contains non-ASCII"
            }
            require(output.size() < maxBytes) {
                "remote capability response header line exceeds bound"
            }
            output.write(value)
        }
    }

    companion object {
        private const val MAX_HEADER_LINE_BYTES =
            8 * 1024
        private const val MAX_HEADER_COUNT = 64
    }
}

class VN97RemoteCapabilityArtifactStore(
    private val root: File,
    private val policy:
        VN97CapabilityArtifactFetchPolicy =
        VN97CapabilityArtifactFetchPolicy(),
    private val transport:
        VN97CapabilityArtifactTransport =
        VN97PinnedHttpsCapabilityTransport(),
) {
    private val rootPath =
        root.toPath().toAbsolutePath().normalize()

    init {
        Files.createDirectories(rootPath)
        require(
            Files.isDirectory(
                rootPath,
                LinkOption.NOFOLLOW_LINKS,
            ) &&
                !Files.isSymbolicLink(rootPath)
        ) {
            "remote capability artifact root must be a real directory"
        }
    }

    @Synchronized
    fun fetch(
        url: String,
    ): VN97FetchedCapabilityArtifact {
        val canonical =
            canonicalCapabilityHttpsUrl(url)
        val bytes = transport.get(
            endpoint = URI(canonical),
            maxResponseBytes =
                policy.maxResponseBytes,
            connectTimeoutMs =
                policy.connectTimeoutMs,
            readTimeoutMs =
                policy.readTimeoutMs,
        )
        require(
            bytes.isNotEmpty() &&
                bytes.size <=
                    policy.maxResponseBytes
        ) {
            "remote capability transport violated response bound"
        }
        val parsed =
            VN97CapabilityPackageParser.parse(
                bytes,
                policy.maxResponseBytes.toLong(),
            )
        require(
            parsed.manifest.kind ==
                "knowledge" &&
                (
                    parsed.manifest.capabilityId ==
                        "knowledge" ||
                        parsed.manifest.capabilityId
                            .startsWith(
                                "knowledge."
                            )
                )
        ) {
            "remote M16 acquisition accepts only knowledge capability packages"
        }
        persist(
            parsed = parsed,
            bytes = bytes,
        )
        return VN97FetchedCapabilityArtifact(
            canonicalUrl = canonical,
            packageSha256 =
                parsed.packageSha256,
            sizeBytes = parsed.size,
            capabilityId =
                parsed.manifest.capabilityId,
            capabilityVersion =
                parsed.manifest.capabilityVersion,
            kind = parsed.manifest.kind,
        )
    }

    fun requireArtifact(
        packageSha256: String,
    ): File {
        require(
            packageSha256.length == 64 &&
                packageSha256.all {
                    it in "0123456789abcdef"
                }
        ) {
            "remote capability artifact digest is invalid"
        }
        val target =
            root.resolve(
                "$packageSha256.vn97cap1"
            )
        require(
            Files.isRegularFile(
                target.toPath(),
                LinkOption.NOFOLLOW_LINKS,
            ) &&
                !Files.isSymbolicLink(
                    target.toPath()
                )
        ) {
            "remote capability artifact does not exist"
        }
        val parsed =
            java.io.RandomAccessFile(
                target,
                "r",
            ).use { file ->
                VN97CapabilityPackageParser.parse(
                    file.channel.map(
                        java.nio.channels.FileChannel
                            .MapMode.READ_ONLY,
                        0L,
                        file.length(),
                    ),
                    policy.maxResponseBytes
                        .toLong(),
                )
            }
        check(
            parsed.packageSha256 ==
                packageSha256
        ) {
            "remote capability artifact content digest changed"
        }
        return target
    }

    private fun persist(
        parsed: VN97ParsedCapabilityPackage,
        bytes: ByteArray,
    ) {
        val target =
            root.resolve(
                "${parsed.packageSha256}.vn97cap1"
            )
        if (
            Files.exists(
                target.toPath(),
                LinkOption.NOFOLLOW_LINKS,
            )
        ) {
            requireArtifact(
                parsed.packageSha256
            )
            return
        }
        val temp =
            Files.createTempFile(
                rootPath,
                ".vn97-artifact-",
                ".tmp",
            )
        try {
            FileOutputStream(
                temp.toFile()
            ).use { output ->
                output.write(bytes)
                output.flush()
                output.fd.sync()
            }
            try {
                Files.createLink(
                    target.toPath(),
                    temp,
                )
            } catch (
                _: FileAlreadyExistsException
            ) {
                requireArtifact(
                    parsed.packageSha256
                )
            }
            requireArtifact(
                parsed.packageSha256
            )
            java.nio.channels.FileChannel
                .open(
                    rootPath,
                    StandardOpenOption.READ,
                )
                .use { it.force(true) }
        } finally {
            Files.deleteIfExists(temp)
        }
    }
}

fun canonicalCapabilityHttpsUrl(
    value: String,
): String {
    require(
        value.isNotEmpty() &&
            value.length <= 2_048 &&
            value == value.trim()
    ) {
        "remote capability URL is missing or too long"
    }
    val input = URI(value)
    require(
        input.scheme?.equals(
            "https",
            ignoreCase = true,
        ) == true
    ) {
        "remote capability URL must use HTTPS"
    }
    require(input.rawUserInfo == null) {
        "remote capability URL must not contain credentials"
    }
    require(input.rawFragment == null) {
        "remote capability URL must not contain a fragment"
    }
    require(
        input.port == -1 ||
            input.port == 443
    ) {
        "remote capability URL port must be 443"
    }
    val rawHost = input.host
    require(!rawHost.isNullOrBlank()) {
        "remote capability URL host is missing"
    }
    val host =
        if (isIpLiteral(rawHost)) {
            rawHost.lowercase()
        } else {
            IDN.toASCII(
                rawHost.lowercase(),
                IDN.USE_STD3_ASCII_RULES,
            )
        }
    require(
        host.isNotBlank() &&
            host.length <= 253 &&
            0.toChar() !in host
    ) {
        "remote capability URL host is invalid"
    }
    val path =
        input.rawPath
            ?.takeIf { it.isNotEmpty() }
            ?: "/"
    require(
        path.startsWith('/') &&
            '\r' !in path &&
            '\n' !in path
    ) {
        "remote capability URL path is invalid"
    }
    val query = input.rawQuery
    require(
        query == null ||
            (
                '\r' !in query &&
                    '\n' !in query
            )
    ) {
        "remote capability URL query is invalid"
    }
    return URI(
        "https",
        null,
        host,
        -1,
        path,
        query,
        null,
    ).toASCIIString()
}

internal fun isGloballyRoutableAddress(
    address: InetAddress,
): Boolean {
    if (
        address.isAnyLocalAddress ||
        address.isLoopbackAddress ||
        address.isLinkLocalAddress ||
        address.isSiteLocalAddress ||
        address.isMulticastAddress
    ) {
        return false
    }
    val bytes = address.address
    return when (address) {
        is Inet4Address -> {
            val a =
                bytes[0].toInt() and 0xff
            val b =
                bytes[1].toInt() and 0xff
            when {
                a == 0 -> false
                a == 10 -> false
                a == 100 &&
                    b in 64..127 -> false
                a == 127 -> false
                a == 169 &&
                    b == 254 -> false
                a == 172 &&
                    b in 16..31 -> false
                a == 192 &&
                    b == 0 -> false
                a == 192 &&
                    b == 168 -> false
                a == 198 &&
                    b in 18..19 -> false
                a == 198 &&
                    b == 51 -> false
                a == 203 &&
                    b == 0 -> false
                a >= 224 -> false
                else -> true
            }
        }
        is Inet6Address -> {
            val first =
                bytes[0].toInt() and 0xff
            val second =
                bytes[1].toInt() and 0xff
            val uniqueLocal =
                first == 0xfc ||
                    first == 0xfd
            val documentation =
                first == 0x20 &&
                    second == 0x01 &&
                    (
                        bytes[2].toInt() and
                            0xff
                    ) == 0x0d &&
                    (
                        bytes[3].toInt() and
                            0xff
                    ) == 0xb8
            val ipv4Mapped =
                bytes.size == 16 &&
                    bytes.take(10).all {
                        it == 0.toByte()
                    } &&
                    bytes[10] ==
                        0xff.toByte() &&
                    bytes[11] ==
                        0xff.toByte()
            !uniqueLocal &&
                !documentation &&
                !ipv4Mapped
        }
        else -> false
    }
}

private fun isIpLiteral(
    host: String,
): Boolean =
    host.contains(':') ||
        host.matches(
            Regex(
                "^[0-9]{1,3}(?:\\.[0-9]{1,3}){3}$"
            )
        )
