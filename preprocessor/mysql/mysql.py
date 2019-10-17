#  Tencent is pleased to support the open source community by making GNES available.
#
#  Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import datetime

from gnes.preprocessor.base import BaseVideoPreprocessor
from gnes.proto import gnes_pb2, blob2array


class MySQLPreprocessor(BaseVideoPreprocessor):

    def __init__(self, user: str,
                 password: str,
                 host: str,
                 port: str,
                 database: str,
                 table_name: str,
                 drop_blob: bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.table_name = table_name
        self.load_db()
        self.drop_blob = drop_blob
        self._NOT_FOUND = None

    def load_db(self):
        import pymysql

        self.connection = pymysql.connect(host=self.host,
                             user=self.user,
                             password=self.password)
        self.cursor = self.connection.cursor()
        self.cursor.execute("CREATE DATABASE IF NOT EXISTS " + self.database)
        self.cursor.execute("USE " + self.database)
        create_table = (
                "CREATE TABLE IF NOT EXISTS %s ("
                "  `doc_id` BIGINT NOT NULL,"
                "  `doc` LONGBLOB,"
                "  `create_time` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,"
                "  PRIMARY KEY (`doc_id`),"
                "  UNIQUE KEY `doc_id` (`doc_id`),"
                "  KEY `create_time_index` (create_time)"
                ") ENGINE=InnoDB") % self.table_name
        self.cursor.execute(create_table)

    def apply(self, docs: 'gnes_pb2.Document', *args, **kwargs) -> None:

        add_iterm = ("INSERT INTO " + self.table_name + "(doc_id, doc, create_time) VALUES (%s, %s, %s)")
        timestamp = datetime.datetime.now()
        timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")

        if self.drop_blob:
            for c in docs.chunks:
                c.ClearField('content')

        data_iterm = (docs.doc_id, docs.SerializeToString(), timestamp)

        try:
            self.cursor.execute(add_iterm, data_iterm)
            self.connection.commit()
        except Exception as e:
            self.logger.warning("Insert failed for doc_id = %s. The reason is: %s" % (docs.doc_id, str(e)))


    def close(self):
        self.cursor.close()
        self.connection.close()

