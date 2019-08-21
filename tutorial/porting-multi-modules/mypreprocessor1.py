from gnes.preprocessor.text.base import BaseTextPreprocessor


class MyPreprocessor1(BaseTextPreprocessor):

    def __init__(self, foo, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.foo = foo

    def apply(self, doc: 'gnes_pb2.Document') -> None:
        super().apply(doc)
        doc.raw_text = doc.raw_bytes.decode().strip()
        doc.raw_text += self.foo
        self.logger.info(doc.raw_text)
