package ai.vn97.app

fun main() {
    fun rejects(block: () -> Unit) {
        check(runCatching(block).isFailure)
    }
    val cleaned = VN97DigitalServices.produce(
        VN97DigitalService.TABLE_CLEANUP,
        "\uFEFFTên,Mô tả\r\n\" Cà phê \",\"ngon, rẻ\"\r\nCà phê,\"ngon, rẻ\"\r\n\r\n",
    )
    check(cleaned.content == "\"Tên\",\"Mô tả\"\r\n\"Cà phê\",\"ngon, rẻ\"\r\n")
    check("bỏ 1 dòng trùng" in cleaned.report)
    check(cleaned.sourceSha256 != cleaned.outputSha256)
    check(cleaned.outputSha256.length == 64)
    val multiline = VN97DigitalServices.produce(
        VN97DigitalService.TABLE_CLEANUP, "A,B\n\"x\"\"y\",\"one\ntwo\"",
    )
    check(multiline.content == "\"A\",\"B\"\r\n\"x\"\"y\",\"one\ntwo\"\r\n")
    val formulas = VN97DigitalServices.produce(
        VN97DigitalService.TABLE_CLEANUP, "A,B\n\t=1+1, -2\n@link,+3",
    )
    check("\"'=1+1\"" in formulas.content)
    check("\"'-2\"" in formulas.content)
    check("4 ô" in formulas.report)
    rejects { VN97DigitalServices.produce(VN97DigitalService.TABLE_CLEANUP, "A,B\n1") }
    rejects { VN97DigitalServices.produce(VN97DigitalService.TABLE_CLEANUP, "A,A\n1,2") }
    rejects { VN97DigitalServices.produce(VN97DigitalService.TABLE_CLEANUP, "A\n\"oops") }
    rejects { VN97DigitalServices.produce(VN97DigitalService.TABLE_CLEANUP, "A\n\"x\"oops") }
    rejects { VN97DigitalServices.produce(VN97DigitalService.TABLE_CLEANUP, "A\nx\"y") }
    rejects {
        VN97DigitalServices.produce(VN97DigitalService.TABLE_CLEANUP, (1..65).joinToString(","))
    }
    rejects {
        VN97DigitalServices.produce(VN97DigitalService.TABLE_CLEANUP, "A\n" + "x\n".repeat(5_001))
    }
    rejects {
        VN97DigitalServices.produce(VN97DigitalService.DOCUMENT_FORMAT, "đ".repeat(70_000))
    }
    rejects { VN97DigitalServices.produce(VN97DigitalService.DOCUMENT_FORMAT, "a\u0000b") }
    rejects { VN97DigitalServices.produce(VN97DigitalService.DOCUMENT_FORMAT, "  ") }
    val text = VN97DigitalServices.produce(
        VN97DigitalService.DOCUMENT_FORMAT, "\r\n  giữ thụt lề \r\n\r\n\r\nViệt Nam  \r\n",
    )
    check(text.content == "  giữ thụt lề\n\nViệt Nam\n")
    check(VN97DigitalServices.produce(VN97DigitalService.DOCUMENT_FORMAT, text.content).content == text.content)
    val page = VN97DigitalServices.produce(
        VN97DigitalService.CATALOG_PAGE, "Tên,Ghi chú\n<script>alert(1)</script>,A&B",
    )
    check("<script>" !in page.content)
    check("&lt;script&gt;" in page.content && "A&amp;B" in page.content)
    check("name=\"viewport\"" in page.content)
    check("https://" !in page.content)
    // Large repeated headings cannot grow an unbounded HTML document.
    rejects {
        VN97DigitalServices.produce(
            VN97DigitalService.CATALOG_PAGE,
            "x".repeat(10_000) + "\n" + (1..200).joinToString("\n"),
        )
    }
    check(VN97ServiceQuote(10, 20).estimatedMarginVnd == -10L)
    rejects { VN97ServiceQuote(-1, 0) }
    rejects { VN97ServiceQuote(Long.MAX_VALUE, 0) }
    println("Digital services: Unicode/CSV/HTML escaping/bounds/quote contracts PASS")
}
