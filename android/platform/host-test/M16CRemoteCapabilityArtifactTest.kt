package ai.vn97.platform

import ai.vn97.runtime.VnJsonNumber
import ai.vn97.runtime.VnStrictJson
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.zip.CRC32
import kotlin.io.path.createTempDirectory

private fun sha(bytes: ByteArray): ByteArray =
    MessageDigest.getInstance("SHA-256").digest(bytes)

private fun hex(bytes: ByteArray): String =
    bytes.joinToString("") {
        "%02x".format(it.toInt() and 0xff)
    }

private fun packageBytes(
    kind: String = "knowledge",
    capabilityId: String = "knowledge.remote.test",
): ByteArray {
    val payload = VnStrictJson.canonical(
        VnStrictJson.objectOf(
            "records" to VnStrictJson.array(
                listOf(
                    VnStrictJson.objectOf(
                        "content" to
                            VnStrictJson.string(
                                "Remote knowledge remains evidence only."
                            ),
                        "title" to
                            VnStrictJson.string(
                                "Remote evidence"
                            ),
                    )
                )
            ),
            "schema" to
                VnStrictJson.string("VN97KN1"),
        )
    ).toByteArray()

    val payloadSha = hex(sha(payload))
    val manifest = VnStrictJson.canonical(
        VnStrictJson.objectOf(
            "capability_id" to
                VnStrictJson.string(capabilityId),
            "capability_version" to
                VnJsonNumber("1"),
            "kind" to
                VnStrictJson.string(kind),
            "schema" to
                VnStrictJson.string("VN97CAP1"),
            "sections" to
                VnStrictJson.array(
                    listOf(
                        VnStrictJson.objectOf(
                            "format" to
                                VnStrictJson.string(
                                    "VN97KN1"
                                ),
                            "index" to
                                VnJsonNumber("1"),
                            "role" to
                                VnStrictJson.string(
                                    "knowledge_records"
                                ),
                            "sha256" to
                                VnStrictJson.string(
                                    payloadSha
                                ),
                            "size" to
                                VnJsonNumber(
                                    payload.size.toString()
                                ),
                        )
                    )
                ),
            "source" to
                VnStrictJson.objectOf(
                    "license" to
                        VnStrictJson.string("test"),
                    "origin" to
                        VnStrictJson.string(
                            "https://example.com/source"
                        ),
                    "source_sha256" to
                        VnStrictJson.string(
                            "77".repeat(32)
                        ),
                ),
        )
    ).toByteArray()

    val sections = listOf(manifest, payload)
    val tableBytes = sections.size * 64
    val payloadOffset = 96 + tableBytes
    val totalSize =
        payloadOffset +
            sections.sumOf { it.size }
    val table =
        ByteBuffer.allocate(tableBytes)
            .order(ByteOrder.LITTLE_ENDIAN)
    var cursor = payloadOffset.toLong()
    sections.forEachIndexed { index, section ->
        table.putInt(
            if (index == 0) 1 else 2
        )
        table.putInt(0)
        table.putLong(cursor)
        table.putLong(section.size.toLong())
        table.put(sha(section))
        table.putLong(0L)
        cursor += section.size
    }
    val content =
        table.array() +
            sections.fold(ByteArray(0)) {
                    acc,
                    section,
                ->
                acc + section
            }
    val header =
        ByteBuffer.allocate(96)
            .order(ByteOrder.LITTLE_ENDIAN)
    header.put("VN97CAP1".toByteArray())
    header.putShort(1.toShort())
    header.putShort(96.toShort())
    header.putInt(0)
    header.putInt(sections.size)
    header.putInt(64)
    header.putLong(96L)
    header.putLong(payloadOffset.toLong())
    header.putLong(totalSize.toLong())
    header.putInt(0)
    header.putInt(0)
    header.put(sha(content))
    val crc = CRC32().apply {
        update(header.array(), 0, 88)
    }.value
    header.putInt(crc.toInt())
    header.putInt(0)
    return header.array() + content
}

private inline fun expectFailure(
    label: String,
    block: () -> Unit,
) {
    check(runCatching(block).isFailure) {
        "expected M16C failure: $label"
    }
}

fun main() {
    check(
        canonicalCapabilityHttpsUrl(
            "HTTPS://Example.COM:443/a/../cap.vn97cap1?x=1"
        ) ==
            "https://example.com/a/../cap.vn97cap1?x=1"
    )
    expectFailure("cleartext") {
        canonicalCapabilityHttpsUrl(
            "http://example.com/cap"
        )
    }
    expectFailure("credentials") {
        canonicalCapabilityHttpsUrl(
            "https://user:pass@example.com/cap"
        )
    }
    expectFailure("fragment") {
        canonicalCapabilityHttpsUrl(
            "https://example.com/cap#fragment"
        )
    }
    expectFailure("non-443") {
        canonicalCapabilityHttpsUrl(
            "https://example.com:8443/cap"
        )
    }

    check(
        isGloballyRoutableAddress(
            java.net.InetAddress.getByName(
                "8.8.8.8"
            )
        )
    )
    listOf(
        "127.0.0.1",
        "10.0.0.1",
        "100.64.0.1",
        "169.254.1.1",
        "192.168.1.1",
        "::1",
        "fd00::1",
        "2001:db8::1",
    ).forEach { value ->
        check(
            !isGloballyRoutableAddress(
                java.net.InetAddress.getByName(
                    value
                )
            )
        ) {
            "expected non-global address: $value"
        }
    }

    val valid = packageBytes()
    var calls = 0
    var seenUrl = ""
    val transport =
        VN97CapabilityArtifactTransport {
                endpoint,
                maxBytes,
                connectMs,
                readMs,
            ->
            calls += 1
            seenUrl = endpoint.toASCIIString()
            check(maxBytes == 16 * 1024 * 1024)
            check(connectMs == 8_000)
            check(readMs == 15_000)
            valid.copyOf()
        }
    val root =
        createTempDirectory(
            "m16c-artifact-"
        ).toFile()
    try {
        val store =
            VN97RemoteCapabilityArtifactStore(
                root = root,
                transport = transport,
            )
        val fetched = store.fetch(
            "https://EXAMPLE.com/cap.vn97cap1"
        )
        check(
            seenUrl ==
                "https://example.com/cap.vn97cap1"
        )
        check(calls == 1)
        check(
            fetched.capabilityId ==
                "knowledge.remote.test"
        )
        check(fetched.kind == "knowledge")
        check(fetched.capabilityVersion == 1L)
        check(
            fetched.packageSha256 ==
                hex(sha(valid))
        )
        val persisted =
            store.requireArtifact(
                fetched.packageSha256
            )
        check(
            persisted.readBytes()
                .contentEquals(valid)
        )

        val malformed =
            VN97RemoteCapabilityArtifactStore(
                root =
                    createTempDirectory(
                        "m16c-bad-"
                    ).toFile(),
                transport =
                    VN97CapabilityArtifactTransport {
                            _,
                            _,
                            _,
                            _,
                        ->
                        "not a package".toByteArray()
                    },
            )
        expectFailure("malformed package") {
            malformed.fetch(
                "https://example.com/bad"
            )
        }

        val nonKnowledge =
            packageBytes(
                kind = "avatar",
                capabilityId = "avatar.remote",
            )
        val nonKnowledgeStore =
            VN97RemoteCapabilityArtifactStore(
                root =
                    createTempDirectory(
                        "m16c-nonknowledge-"
                    ).toFile(),
                transport =
                    VN97CapabilityArtifactTransport {
                            _,
                            _,
                            _,
                            _,
                        ->
                        nonKnowledge
                    },
            )
        expectFailure("non-knowledge package") {
            nonKnowledgeStore.fetch(
                "https://example.com/avatar"
            )
        }

        val wire = fetched.wireResult()
        val parsed =
            VN97FetchedCapabilityArtifact
                .parseWireResult(wire)
        check(
            parsed.packageSha256 ==
                fetched.packageSha256
        )
        check(
            parsed.sizeBytes ==
                fetched.sizeBytes
        )
    } finally {
        root.deleteRecursively()
    }

    println(
        "M16C remote capability artifact contracts: PASS"
    )
}
