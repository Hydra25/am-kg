from src.pdf_parser.grobid_parser import parse_tei

TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt><title level="a" type="main">LPBF of Ti-6Al-4V</title></titleStmt>
<sourceDesc><biblStruct><analytic><author><persName><forename>A.</forename><surname>Author</surname></persName></author></analytic>
<monogr><title level="j">Additive Manufacturing</title><imprint><date type="published" when="2021-03-01"/></imprint></monogr>
<idno type="DOI">10.1016/j.addma.2021.000</idno></biblStruct></sourceDesc></fileDesc>
<profileDesc><abstract><p><s coords="1,10,10,100,10">Laser power reduced porosity.</s></p></abstract></profileDesc></teiHeader>
<text><body><div><head n="2">Materials and methods</head><p><s coords="2,1,1,1,1">Ti-6Al-4V was built by LPBF.</s></p></div>
<div><head n="3">Results</head><p><s coords="4,1,1,1,1">Porosity was 0.2 %.</s></p></div></body></text></TEI>"""


def test_parse_tei():
    meta, secs, sents, tables = parse_tei(TEI, "p1", "x.pdf")
    assert meta.title == "LPBF of Ti-6Al-4V" and meta.year == 2021 and meta.doi.startswith("10.1016")
    assert meta.journal == "Additive Manufacturing" and meta.authors == ["A. Author"]
    assert [s.section_type for s in secs] == ["abstract", "materials_methods", "results"]
    assert sents[1].page == 2 and sents[2].page == 4
